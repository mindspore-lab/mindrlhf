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
from mindspore.nn.cell import Cell
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
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved, _MicroBatch
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
# try:
from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig, AttentionMask

# except ImportError as e:
#     print("Import ERROR, expect mindformers to be installed. "
#           "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
#     print("Now exit the program.")
#     exit(1)

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
from generator import GeneratorMixin
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.models.bloom import BloomLMHeadModel, BloomConfig
from mindformers.models.pangualpha import PanguAlphaHeadModel, PanguAlphaConfig
from mindformers.modules.layers import Linear

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)

@dataclass
class PPOConfig:
    """
    PPO config class which defines the model size
    """
    epochs: int = 2
    total_steps: int = 100000
    batch_size: int = 1
    checkpoint_interval = 10000
    eval_interval: int = 200

    optimizer: str = 'adamw'
    lr: float = 5.0e-5
    betas = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.01

    sceduler_name: str = 'cosine_annealing'
    T_max: int = 100000
    eta_min: float = 5.0e-6

    num_rollouts: int = 8
    chunk_size: int = 4
    ppo_epochs: int = 4
    init_kl_coef: float = 0.1
    kl_coef: float = 0.02
    target: float = 6.0
    horizon: int = 10000
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 1.0
    pretrain_coef: float = 0.9
    scale_reward: bool = None
    ref_mean: bool = False
    ref_std: bool = False
    gen_experience_kwargs: bool = False

    seq_length: int = 8192
    max_prompt_length: int = 4096
    max_decode_length: int = 4096
    vocab_size: int = 40000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expand_ratio: int = 4
    post_layernorm_residual: bool = False
    dropout_rate: float = 0.1
    dtype: mstype = mstype.float16
    compute_dtype: mstype = mstype.float16
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float32
    eos_token_id = 2
    pad_token_id = 0
    repetition_penalty = 1
    top_k = 1
    top_p = 0.95
    is_encoder_decoder = False
    do_sample = False


class LogprobsOfLabels(nn.Cell):
    def __init__(self):
        super(LogprobsOfLabels, self).__init__()
        self.cast = ops.Cast()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gatherd = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)
    def construct(self, logits, labels):
        labels = self.cast(labels, mindspore.int32)
        logprobs = self.log_softmax(logits)
        logprobs_labels = self.gatherd(logprobs, -1, self.unsqueeze(labels, -1))
        return self.squeeze(logprobs_labels)

class ProcessLogits(nn.Cell):
    def __init__(self):
        super(ProcessLogits, self).__init__()
        self.e = Tensor(np.e)
        self.gather = P.Gather()
        self.logsoftmax = P.LogSoftmax()
        self.reshape = P.Reshape()

    def construct(self, logits, current_index=None):
        """Process the logits"""
        if current_index is not None:
            index = current_index.view(-1,)
            if len(logits.shape) == 3:
                logits = self.reshape(logits, (logits.shape[0]*logits.shape[1], -1))
            logits = self.gather(logits, index, 0)
        outputs = self.logsoftmax(logits)
        outputs = F.tensor_pow(self.e, outputs)
        return outputs

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

    '''def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁'''

class FixedKLController(nn.Cell):
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = Tensor([kl_coef, ])

    def construct(self, current, n_steps):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        return Tensor(0.0, mstype.float16)

    '''def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass'''


class CausalLMHydraWithValueHead(nn.Cell):
    """
    CausalLMHydraWithValueHead
    """

    def __init__(self, model_config, ppo_config, is_training=True):
        super(CausalLMHydraWithValueHead, self).__init__()
        if not is_training:
            model_config.dropout_rate = 0.0
        self.model_config = model_config
        self.ppo_config = ppo_config

        if isinstance(model_config, PanguAlphaConfig):
            self.model_type = 'pangu'
        elif isinstance(model_config, BloomConfig):
            self.model_type = 'bloom'
        else:
            raise NotImplementedError("only support pangu and bloom")
        print("sft model_type: ", self.model_type)
        
        if self.model_type == 'pangu':
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
            self.lm_head = self.model.head
        elif self.model_type == 'bloom':
            self.model = BloomLMHeadModel(model_config)
            self.backbone = self.model.transformer
            self.lm_head = self.model.head
        
        self.lm_head.pipeline_stage = model_config.parallel_config.pipeline_stage - 1
        dp = model_config.parallel_config.data_parallel
        mp = model_config.parallel_config.model_parallel

        self.vocab_size = model_config.vocab_size
        self.chunk_size = ppo_config.chunk_size
        self.seq_length = ppo_config.seq_length
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.all_ones_attention_mask = Tensor(np.ones((1, 1, self.seq_length)), mstype.float32)

        self.squeeze = P.Squeeze(axis=-1).shard(((dp, 1, 1), ))
        self.squeeze_no_shard = P.Squeeze(axis=-1).shard(((1, 1, 1), ))
        self.unsqueeze = P.ExpandDims()
        self.reshape = P.Reshape()
        self.e = Tensor(np.e)
        self.gather = P.Gather().shard(((dp, mp),
                                        (1, ), ))
        self.strided_slice_1 = P.StridedSlice().shard(((dp, 1, mp), ))
        self.strided_slice_2 = P.StridedSlice().shard(((dp, 1), ))
        self.gatherd = P.GatherD()
        self.logsoftmax_1 = P.LogSoftmax().shard(((1, 1, 1), ))
        self.logsoftmax_2 = P.LogSoftmax().shard(((dp, 1), ))

        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.sum = P.ReduceSum().shard(((dp, mp),))
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(((dp, mp),))
        self.sub = P.Sub().shard(((dp, mp), (dp, 1)))
        self.exp = P.Exp().shard(((dp, mp),))
        self.div = P.RealDiv().shard(((dp, mp), (dp, 1)))
        self.onehot = P.OneHot().shard(((dp, mp), (), ()))
        self.log = P.Log().shard(((dp, mp),))
        self.pow = P.Pow().shard(((dp, 1), ()))
        self.argmax_no_shard = P.Argmax(-1).shard(((1, 1), ))
        self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
        self.sub_shard = P.Sub().shard(((), (1, 1, 1)))

    def process_logits(self, logits, current_index=None, is_first_iteration=False, use_past=False):
        logits = logits.reshape(-1, logits.shape[-1])
        if use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1,)
            logits = self.gather(logits, index, 0)
        outputs = self.logsoftmax_2(logits)
        outputs = self.pow(outputs, Tensor(np.e, mstype.float32))
        return outputs

    def process_logits2(self, logits, current_index=None, is_first_iteration=False, use_past=False):
        logits = logits.reshape(-1, logits.shape[-1])
        if use_past and not is_first_iteration:
            logits = logits
        elif current_index is not None:
            index = current_index.view(-1,)
            logits = self.gather(logits, index, 0)
        top_token_id = self.argmax_no_shard(logits)
        top_token_id = top_token_id.view(-1, 1)
        return top_token_id

    def logprobs_of_labels(self, logits, samples, batch_size, seq_length):
        logits = logits[:, :-1, :]
        samples = samples[:, 1:]
        logprobs = self.logsoftmax_1(logits)
        logprobs = self.squeeze_no_shard(self.gather(logprobs, -1, self.unsqueeze(samples, -1)))

        ''' This implementation has overflow problem
        samples = self.strided_slice_2(samples, (0, 1), (batch_size, self.seq_length), (1, 1))
        logits = self.reshape(logits, (batch_size*seq_length, -1))
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        logprobs = self.log(softmax_result)
        logprobs = self.reshape(logprobs, (batch_size, seq_length, -1))

        logprobs = self.strided_slice_1(logprobs, (0, 0, 0), (batch_size, -1, self.vocab_size), (1, 1, 1))
        logprobs_labels = self.squeeze_no_shard(self.gatherd(logprobs, -1, self.unsqueeze(samples, -1)))
        return logprobs_labels'''
        
        return logprobs

    def construct(self,
                  # inputs for the llm
                  input_ids, 
                  input_position=None, 
                  position_ids=None,
                  attention_mask=None,
                  init_reset=True, 
                  batch_valid_length=None,
                  # inputs for `process_logits`
                  is_first_iteration=False, 
                  use_past=False,
                  # inputs for choosing the output branch
                  samples=None,
                  return_full_logit=False):
        batch_size, seq_length = input_ids.shape
        if self.model_type == 'pangu':
            if self.model.phase == 'train':
                seq_length = seq_length - 1
                tokens = self.model.slice(input_ids, (0, 0), (batch_size, seq_length), (1, 1))
            else:
                tokens = input_ids
            input_mask = F.cast(self.model.not_equal(tokens, self.model.pad_token_id),
                                mstype.float32)
            if attention_mask is None:
                attention_mask = self.model.get_attention_mask(input_mask)
            else:
                attention_mask = self.model.cast(attention_mask, mstype.float32)
                attention_mask = self.model.slice2(attention_mask, (0, 0, 0),
                                                    (batch_size, seq_length, seq_length),
                                                    (1, 1, 1))

            if input_position is None:
                input_position = F.tuple_to_array(F.make_range(seq_length))
                input_position = self.model.expand(input_position, 0)
                if batch_size == 1:
                    input_position = F.reshape(input_position, (1, seq_length))
                else:
                    input_position = self.model.tile(input_position, (batch_size, 1))
            else:
                if self.model.phase == 'train':
                    input_position = self.model.slice(input_position, (0, 0), (batch_size, seq_length), (1, 1))

            # [batch_size, seq_length, vocab_size]
            output_states, embedding_table = self.backbone(tokens, input_position, attention_mask,
                                                           init_reset, batch_valid_length)
        else:
            if self.model.phase == "train":
                tokens = self.model.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
            else:
                tokens = input_ids

            if self.model.use_past:
                input_mask = self.model.input_mask_all_ones
            else:
                input_mask = self.model.not_equal(tokens, self.model.eos_token_id).astype(mstype.float32)

            output_states, embedding_table = self.backbone(tokens, input_mask, init_reset, batch_valid_length)
        
        logits = self.lm_head(output_states, embedding_table)
        logits = self.reshape(logits, (batch_size, seq_length, -1))

        # This model is used in three places, generate, make_experience, and ppo
        # to reduce memory, we return the required output in different places

        # used in inference of make_experience and ppo
        if samples is not None:
            logprobs_labels = self.logprobs_of_labels(logits, samples, batch_size, seq_length)
            return logprobs_labels
        
        # used in generate
        elif return_full_logit == False:
            outputs = self.process_logits2(logits, input_position, is_first_iteration, use_past)
            return outputs

        # used in pretrain loss
        else:
            return logits


class Sampler(nn.Cell):
    def __init__(self):
        super(Sampler, self).__init__()
        self.shape = P.Shape()
        self.zeros = P.Zeros()
        self.top_k = P.TopK(sorted=True)

    def construct(self, log_probs, batch_size, top_k, repetition_penalty, frequency_list):
        vocab_size = self.shape(log_probs)[-1]
        if repetition_penalty != 1 and frequency_list is None:
            frequency_list = self.zeros((vocab_size, ), mindspore.float32)
        log_probs_revised = log_probs.reshape(batch_size, vocab_size)
        if repetition_penalty != 1:
                log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                    (frequency_list > 0) * repetition_penalty
        logits = F.pow(Tensor(np.e), log_probs_revised)
        probs, p_args = self.top_k(logits, top_k)

        norm = probs.sum(-1, keepdims=True)
        avg = F.broadcast_to(Tensor(1/top_k), probs.shape).to(mstype.float32)
        probs = F.select(norm==0, avg, probs/norm)
        return probs, p_args


class PPO_model(nn.Cell, GeneratorMixin):
    """
    PPO_model
    """

    def __init__(self, ppo_config, policy_model, critic_model, opt, is_training=True):
        super(PPO_model, self).__init__()
        self.ppo_config = ppo_config
        self.pretrain_coef = self.ppo_config.pretrain_coef
        self.opt = opt
        self.use_past = opt.use_past
        print("use_past: ", self.use_past)
        self.pad_token_id = Tensor(ppo_config.pad_token_id, mstype.int32)
        self.policy_model = policy_model
        self.critic_model = critic_model
        # self.inference_wrapper = PipelineCell_inference(self.policy_model, self.opt.inference_micro_size)
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

        if ppo_config.target is not None:
            self.kl_ctl = AdaptiveKLController(ppo_config.init_kl_coef, ppo_config.target, ppo_config.horizon)
        else:
            self.kl_ctl = FixedKLController(ppo_config.init_kl_coef)

        self.logprobs_of_labels = LogprobsOfLabels()
        self.process_logits = ProcessLogits()
        self.sampler = Sampler()
        self.slice = P.StridedSlice().shard(((self.policy_model.model_config.parallel_config.data_parallel, 1),))
        self.pretrain_loss = CrossEntropyLoss(self.policy_model.model_config.parallel_config.dp_mp_config)
        self.not_equal = P.NotEqual().shard(((self.policy_model.model_config.parallel_config.data_parallel, 1), ()))
        self.reduce_mean = P.ReduceMean()
        self.depend = P.Depend()

    def construct(self,
                  query_tensors,
                  response_tensors,
                  logprobs,
                  values,
                  rewards,
                  attention_mask,
                  advantages,
                  returns,
                  pretrain_ids,
                  loss_mask):
        """
        """
        # print("data: ", query_tensors, response_tensors, logprobs, values, rewards, attention_mask)
        old_logprobs = logprobs
        old_rewards = rewards
        old_values = values
        response_length = F.shape(old_rewards)[1]
        tokens = response_tensors

        '''input_mask = F.cast(F.not_equal(tokens, self.pad_token_id), mstype.float32)
        bs, seq_length = F.shape(response_tensors)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        attention_mask_pangu = self.get_attention_mask(input_mask)'''

        # attention_mask = self.attention_mask
        logprobs = self.policy_model(tokens, samples=tokens)
        tokens = self.depend(tokens, logprobs)
        values_pred = self.critic_model(tokens)

        start = F.shape(query_tensors)[1] - 1
        end = start + response_length
        
        '''logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
            )
        values_pred = values_pred[:, start:end]
        mask = attention_mask[:, start:end]'''
        
        logprobs = logprobs[:, start:end]
        values_pred = values_pred[:, start:end]
        mask = attention_mask[:, start:end]

        # calculate pretrain loss
        batch_size, seq_length = response_tensors.shape
        pretrain_tokens = self.slice(pretrain_ids, (0, 0), (batch_size, -1), (1, 1))
        logits = self.policy_model(pretrain_tokens, return_full_logit=True)
        logits = P.Reshape()(logits, (batch_size * seq_length, -1))
        labels = self.slice(pretrain_ids, (0, 1), (batch_size, seq_length + 1), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        pretrain_loss = self.pretrain_loss(logits, labels, loss_mask)

        # calculate value loss and policy loss
        vf_loss, pg_loss, approx_kl = self.get_vfloss_and_pgloss(
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

        loss = self.pretrain_coef * pretrain_loss + (1-self.pretrain_coef) * pg_loss + self.vf_coef * vf_loss
        return loss

    def post_backward_callback(self):
        return self.kl_ctl(self.approx_kl, Tensor([self.ppo_config.batch_size, ]))

    
    def get_vfloss_and_pgloss(
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
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
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

        log_ratio = (logprobs - old_logprobs) * mask

        ratio = self.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        
        approx_kl = self.reduce_mean((ratio - 1) - log_ratio)
        approx_kl = self.stop_grad(approx_kl)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * ops.clip_by_value(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
        )
        pg_loss = self.reduce_sum(self.max(pg_loss1, pg_loss2) * mask) / n

        return vf_loss, pg_loss, approx_kl