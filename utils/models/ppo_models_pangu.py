# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""PPO model"""
import datetime
import json
import glob
import os
import math
import copy
import numpy as np
from dataclasses import dataclass

import mindspore
from mindspore import context, Tensor, Parameter, ops
from mindspore.train.model import Model
from mindspore.common.initializer import TruncatedNormal
import mindspore.communication.management as D
import mindspore.ops.functional as F
from mindspore.ops import ReduceOp
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
try:
    from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig, AttentionMask

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)

from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha import PanGUAlphaWithLoss, PanguAlphaModel, PanguAlpha_Model, PanGuHead
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from src.callbacks import EvalCallBack, LossCallBack
from src.metrics import PPLMetric
from mindspore.ops.operations._inner_ops import Send, Receive
from utils.generator_pangu import GeneratorMixin
# from .ppo_models import AdaptiveKLController

class AdaptiveKLController(nn.Cell):
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        super(AdaptiveKLController, self).__init__()
        self.value = Parameter([init_kl_coef,], requires_grad=False)
        self.target = Tensor(target)
        self.horizon = Tensor(horizon)
        self.div = P.Div()
        self.add = P.Add()

    def construct(self, current, n_steps):
        proportional_error = self.add(self.div(current, self.target), Tensor(-1, mstype.float32))
        proportional_error = proportional_error.clip(-0.2, -0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult
        return self.value


class PanguWithValueHead(nn.Cell):
    """
    PanguWithValueHead
    """

    def __init__(self, model_config, ppo_config, is_training=True):
        super(PanguWithValueHead, self).__init__()
        if not is_training:
            model_config.dropout_rate = 0.0
        self.model_config = model_config
        self.ppo_config = ppo_config
        copied_parallel_config = copy.deepcopy(model_config.parallel_config)

        if copied_parallel_config.pipeline_stage > 1:
            copied_parallel_config.vocab_emb_dp = False

        self.backbone = PanguAlpha_Model(model_config)
        self.lm_head = PanGuHead(hidden_size=model_config.hidden_size,
                                 parallel_config=copied_parallel_config)
        self.lm_head.pipeline_stage = model_config.parallel_config.pipeline_stage - 1

        self.vocab_size = model_config.vocab_size
        self.cast = P.Cast()
        self.shape = P.Shape()
        
        self.v_head0 = nn.Dense(model_config.hidden_size,
                                2*model_config.hidden_size,
                                weight_init=TruncatedNormal(0.02),
                                activation="relu",
                                has_bias=True).to_float(mstype.float16)
        self.v_head2 = nn.Dense(2*model_config.hidden_size,
                                1,
                                weight_init=TruncatedNormal(0.02),
                                has_bias=True).to_float(mstype.float16)
        self.v_head0.pipeline_stage = model_config.parallel_config.pipeline_stage - 1
        self.v_head2.pipeline_stage = model_config.parallel_config.pipeline_stage - 1

        self.squeeze = P.Squeeze(axis=-1)
        

    def construct(self, input_ids, input_position, attention_mask, init_reset=True, batch_valid_length=None):
        output_states, word_table = self.backbone(input_ids, input_position, attention_mask, init_reset, batch_valid_length)
        batch_size, seq_length = self.shape(input_ids)
        lm_logits = self.lm_head(output_states, word_table)
        lm_logits = P.Reshape()(lm_logits, (batch_size, seq_length, -1))
        values = self.v_head0(output_states)
        values = self.v_head2(values)
        values = P.Reshape()(values, (batch_size, seq_length))
        return lm_logits, values


class SRNet(nn.Cell):
    def __init__(self, stage_id, rank_id, stage_num, device_num, bsz, seq_length, vocab_size):
        super(SRNet, self).__init__()
        self.device_num = device_num
        self.stage_num = stage_num
        stage_device_num = int(device_num/stage_num)
        self.stage_device_num = stage_device_num
        self.rank = rank_id
        self.send_list = []
        self.temp = Tensor(0, mstype.float32)
        for i in range(stage_num-1):
            self.send_list.append(Send(i, rank_id-(stage_num-i-1)*stage_device_num)) 

        self.recv = Receive(stage_id, rank_id+(stage_num-stage_id-1)*stage_device_num, [bsz, seq_length, vocab_size], mstype.float16)
        print("SR: ", self.device_num, self.stage_num, self.stage_device_num, self.rank, self.send_list, self.recv)
    def construct(self, x):
        if self.device_num - self.stage_device_num <= self.rank:
            for i in range(self.stage_num-1):
                self.send_list[i](x[0])
            return x[0]
        else:
            return self.recv(self.temp)

class SRNet_1(nn.Cell):
    def __init__(self, stage_id, rank_id, stage_num, device_num, bsz, seq_length, vocab_size):
        super(SRNet_1, self).__init__()
        self.device_num = device_num
        self.stage_num = stage_num
        stage_device_num = int(device_num/stage_num)
        self.stage_device_num = stage_device_num
        self.rank = rank_id
        self.send_list = []
        self.temp = Tensor(0, mstype.float32)
        for i in range(stage_num-1):
            self.send_list.append(Send(i+device_num, rank_id-(stage_num-i-1)*stage_device_num)) 

        self.recv = Receive(stage_id+device_num, rank_id+(stage_num-stage_id-1)*stage_device_num, [bsz, seq_length, 1], mstype.float16)
        print("SR: ", self.device_num, self.stage_num, self.stage_device_num, self.rank, self.send_list, self.recv)
    def construct(self, x):
        if self.device_num - self.stage_device_num <= self.rank:
            for i in range(self.stage_num-1):
                self.send_list[i](x[1])
            return x[1]
        else:
            return self.recv(self.temp)

class SRNet_2(nn.Cell):
    def __init__(self, stage_id, rank_id, stage_num, device_num, bsz, seq_length, vocab_size):
        super(SRNet_2, self).__init__()
        self.device_num = device_num
        self.stage_num = stage_num
        stage_device_num = int(device_num/stage_num)
        self.stage_device_num = stage_device_num
        self.rank = rank_id
        self.send_list = []
        self.temp = Tensor(0, mstype.float32)
        for i in range(stage_num-1):
            self.send_list.append(Send(i+device_num*2, rank_id-(stage_num-i-1)*stage_device_num)) 

        self.recv = Receive(stage_id+device_num*2, rank_id+(stage_num-stage_id-1)*stage_device_num, [bsz, 1], mstype.float16)
        print("SR: ", self.device_num, self.stage_num, self.stage_device_num, self.rank, self.send_list, self.recv)
    def construct(self, x):
        if self.device_num - self.stage_device_num <= self.rank:
            for i in range(self.stage_num-1):
                self.send_list[i](x)
            return x
        else:
            return self.recv(self.temp)


class PPO_model(nn.Cell, GeneratorMixin):
    """
    PPO_model
    """

    def __init__(self, ppo_config, policy_model, opt, is_training=True):
        super(PPO_model, self).__init__()
        self.ppo_config = ppo_config
        self.opt = opt
        self.pad_token_id = Tensor(ppo_config.pad_token_id, mstype.int32)
        self.policy_model = policy_model
        self.stack = P.Stack(axis=1)
        self.allreduce_sum = P.AllReduce(ReduceOp.SUM)
        self.reduce_sum = P.ReduceSum(keep_dims=False) 

        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.rsqrt = P.Rsqrt()
        self.concat = P.Concat(1)

        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)

        self.cliprange_value = ppo_config.cliprange_value
        self.cliprange = ppo_config.cliprange
        self.vf_coef = ppo_config.vf_coef
        self.max = P.Maximum()
        self.cast = P.Cast()
        self.exp = P.Exp()
        self.stop_grad = P.StopGradient()
        self.gamma = 1
        self.lam=0.95
        self.size = P.Size()
        self.sum_and_count = Tensor([[1,1]])
        # self.attention_mask = Tensor(np.ones((1, 1024)), mstype.float32)
        
        self.approx_kl = Parameter([0.0, ], requires_grad=False)
        self.get_attention_mask = AttentionMask(ppo_config.seq_length)

        device_num = D.get_group_size()
        rank_id = D.get_rank()
        stage_num = self.policy_model.model_config.parallel_config.pipeline_stage
        per_stage_num = int(device_num / stage_num)
        stage_id = int(rank_id / per_stage_num)
        
        bsz = self.ppo_config.chunk_size
        seq_length = self.ppo_config.seq_length
        vocab_size = self.policy_model.model_config.vocab_size
        self.sr_net = SRNet(stage_id, rank_id, stage_num, device_num, bsz, seq_length, vocab_size)
        self.kl_ctl = AdaptiveKLController(ppo_config.init_kl_coef, ppo_config.target, ppo_config.horizon)

    def construct(self, query_tensors, response_tensors, logprobs, values, rewards, attention_mask):
        """
        """
        old_logprobs = logprobs
        old_rewards = rewards
        old_values = values
        response_length = F.shape(old_rewards)[1]
        advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, response_length)
        tokens = response_tensors

        input_mask = F.cast(F.not_equal(tokens, self.pad_token_id), mstype.float32)
        bs, seq_length = F.shape(response_tensors)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        attention_mask_pangu = self.get_attention_mask(input_mask)

        # attention_mask = self.attention_mask
        logits, values_pred, _ = self.policy_model(tokens, input_position, attention_mask_pangu)

        values_pred = values_pred[:, :-1]
        logprobs = self.logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

        start = F.shape(query_tensors)[1] - 1
        end = start + response_length
        
        logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
            )

        loss, approx_kl = self.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        approx_kl = ops.Reshape()(approx_kl, (1, ))
        self.approx_kl = approx_kl
        return loss

    def post_backward_callback(self):
        return self.kl_ctl(self.approx_kl, Tensor([self.ppo_config.batch_size, ]))

    def get_advantages_and_returns(
        self,
        values,
        rewards,
        response_length,
        use_whitening: bool = True,
    ):

        lastgaelam = 0
        advantages_reversed = []

        for k in range(response_length):
            t = response_length-k-1
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = self.stack(advantages_reversed[::-1])

        
        returns = advantages + values
        if use_whitening:
            advantages = self.whiten(advantages)

        return self.stop_grad(advantages), returns

    def get_global_statistics(self, xs):
        """
        Computes element-wise mean and variance of the tensor across processes
        """
        sum_and_count = Tensor([self.reduce_sum(xs), self.size(xs)])
        sum_and_count = self.allreduce_sum(sum_and_count)
        global_sum, count = sum_and_count
        global_mean = global_sum / count

        sum_var = self.reduce_sum((xs - global_mean) ** 2)
        sum_var = self.allreduce_sum(sum_var)
        global_var = sum_var / count
        return global_mean, global_var, count

    def whiten(self, xs, shift_mean=True, distributed=False):
        """Whitens values"""
        if distributed:
            mean, var, _ = self.get_global_statistics(xs)
        else:
            mean = self.reduce_mean(xs)

            var = xs.var()

        whitened = (xs - mean) * self.rsqrt(var + 1e-8)
        if not shift_mean:
            whitened += mean
        return whitened

    def logprobs_of_labels(self, logits, labels):
        """Log probabilities of the labels

        These are calculated from the logits."""
        logprobs = self.log_softmax(logits)
        logprobs_labels = self.gather(logprobs, -1, self.unsqueeze(labels, -1))
        return self.squeeze(logprobs_labels)
    
    def loss(
        self,
        logprobs,
        values,
        old_logprobs,
        old_values,
        advantages,
        returns,
        mask,
    ):
        """PPO objective function.
        """
        values_clipped = ops.clip_by_value(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2

        vf_loss = 0.5 * self.reduce_sum(self.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = self.reduce_sum(self.cast((vf_loss2 > vf_loss1), mstype.float16) * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask

        ratio = self.exp(log_ratio)
        
        approx_kl = self.reduce_mean((ratio - 1) - log_ratio)
        approx_kl = self.stop_grad(approx_kl)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * ops.clip_by_value(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = self.reduce_sum(self.max(pg_loss1, pg_loss2) * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss

        return loss, approx_kl