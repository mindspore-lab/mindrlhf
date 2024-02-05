# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
# limitations under the License
# ============================================================================
"""PPO model"""
import numpy as np

import mindspore
from mindspore import Tensor, Parameter, ops
import mindspore.ops.functional as F
from mindspore.ops import ReduceOp
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindformers.modules.transformer import AttentionMask
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.layers import Linear
from mindrlhf.utils.generator import GeneratorMixin
from .base_model import BaseModel

__all__ = ['PPO_model', 'LogprobsOfLabels', 'ProcessLogits', 'FixedKLController',
           'CausalLMHydraWithValueHead', 'Sampler',]


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


class FixedKLController(nn.Cell):
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = Tensor([kl_coef, ])

    def construct(self, current, n_steps):
        """Returns updated KL coefficient, βₜ₊₁.
        """
        return Tensor(0.0, mstype.float16)


class CausalLMHydraWithValueHead(BaseModel):
    """
    CausalLMHydraWithValueHead
    """

    def __init__(self, model_config, ppo_config, is_training=True):
        super(CausalLMHydraWithValueHead, self).__init__()
        if not is_training:
            model_config.dropout_rate = 0.0
        self.model_config = model_config
        self.ppo_config = ppo_config
        self.select_actor_model(model_config)
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
        self.argmax = P.Argmax(-1).shard(((dp, mp),))
        self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
        self.sub_shard = P.Sub().shard(((), (1, 1, 1)))
        self.add_shard = P.Add().shard(((1, 1, 1), ()))

        self.minus_one = Tensor([-1], mstype.int32)
        self.v_head0 = Linear(self.ppo_config.hidden_size, 2*self.ppo_config.hidden_size,
                              activation='relu', has_bias=True)
        self.v_head1 = Linear(2*self.ppo_config.hidden_size, 1, has_bias=True)

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
        logprobs = self.squeeze_no_shard(self.gatherd(logprobs, -1, self.unsqueeze(samples, -1)))

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
                  return_full_logit=False,
                  return_value=False):
        batch_size, seq_length = input_ids.shape
        if self.model_type == 'pangu':
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
            logits_2d = self.lm_head(output_states, embedding_table)
        elif self.model_type == 'baichuan2_7b':
            tokens = input_ids
            output_states = self.backbone(tokens, batch_valid_length)
            logits_2d = self.lm_head(output_states)
        elif self.model_type == 'baichuan2_13b':
            tokens = input_ids
            output_states = self.backbone(tokens, batch_valid_length)
            logits_2d = self.lm_head(output_states)
        elif self.model_type == 'gpt2':
            tokens = input_ids
            if attention_mask is None:
                attention_mask = self.model.not_equal(input_ids, self.model.pad_token_id)
            attention_mask = self.cast(attention_mask, mstype.float32)
            attention_mask = self.model.get_attention_mask(attention_mask)
            if not self.model.is_first_iteration:
                attention_mask = self.model.tile(self.model.all_ones_attention_mask, (batch_size, 1, 1))
            #
            # # [batch_size, seq_length, vocab_size]
            output_states, embedding_table = self.backbone(
                tokens, attention_mask, input_position=input_position,
                init_reset=init_reset, batch_valid_length=batch_valid_length)
            logits_2d = self.lm_head(output_states, embedding_table)
        elif self.model_type == 'llama':
            if self.model.phase == "train":
                tokens = self.model.slice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
            else:
                tokens = input_ids
            output_states = self.backbone(tokens, input_position, init_reset, batch_valid_length)
            logits_2d = self.lm_head(output_states)
        else:
            if self.model.phase == "train":
                tokens = self.model.stridedslice(input_ids, (0, 0), (batch_size, seq_length - 1), (1, 1))
            else:
                tokens = input_ids

            if self.model.use_past:
                input_mask = self.model.input_mask_all_ones
            else:
                input_mask = self.model.not_equal(tokens, self.model.pad_token_id).astype(mstype.float32)

            output_states, embedding_table = self.backbone(tokens, input_mask, init_reset, batch_valid_length)
            logits_2d = self.lm_head(output_states, embedding_table)

        # if self.model_type == 'baichuan' or self.model_type == 'llama':
        #     logits_2d = self.lm_head(output_states)
        # else:
        #     logits_2d = self.lm_head(output_states, embedding_table)

        logits = self.reshape(logits_2d, (batch_size, seq_length, -1))

        # used in inference of make_experience and ppo
        if samples is not None:
            logprobs_labels = self.logprobs_of_labels(logits, samples, batch_size, seq_length)
            if return_value == True:
                value = self.v_head1(self.v_head0(output_states))
                value = self.reshape(value, (batch_size, seq_length))
                return logprobs_labels, value
            return logprobs_labels

        # used in generate
        elif return_full_logit == False:
            outputs = self.process_logits2(logits, input_position, is_first_iteration, use_past)
            return outputs

        # used in pretrain loss
        else:
            logits = self.add_shard(logits, 0)
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
        probs = F.select(norm == 0, avg, probs/norm)
        return probs, p_args


class PPO_model(nn.Cell, GeneratorMixin):
    """
    PPO_model
    """

    def __init__(self, ppo_config, policy_model, critic_model):
        super(PPO_model, self).__init__()
        self.ppo_config = ppo_config
        self.pretrain_coef = self.ppo_config.pretrain_coef
        self.use_past = self.ppo_config.use_past
        self.pad_token_id = Tensor(ppo_config.pad_token_id, mstype.int32)
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.stack = P.Stack(axis=1)
        self.allreduce_sum = P.AllReduce(ReduceOp.SUM)
        self.reduce_sum = P.ReduceSum(keep_dims=False)

        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.rsqrt = P.Rsqrt()
        self.concat = P.Concat(1)

        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        self.unsqueeze = P.ExpandDims().shard(((1, 1),))
        self.squeeze = P.Squeeze(axis=-1)

        self.cliprange_value = ppo_config.cliprange_value
        self.cliprange = ppo_config.cliprange
        self.vf_coef = ppo_config.vf_coef
        self.max = P.Maximum()
        self.cast = P.Cast()
        self.exp = P.Exp()
        self.stop_grad = P.StopGradient()
        self.gamma = 1
        self.lam = 0.95
        self.size = P.Size()
        self.sum_and_count = Tensor([[1, 1]])
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
                  advantages,
                  returns,
                  pretrain_ids,
                  loss_mask,
                  attention_mask):
        old_logprobs = logprobs
        old_rewards = rewards
        old_values = values
        response_length = F.shape(old_rewards)[1]
        tokens = response_tensors
        logprobs = self.policy_model(tokens, samples=tokens)
        tokens = self.depend(tokens, logprobs)
        values_pred = self.critic_model(tokens)

        start = F.shape(query_tensors)[1] - 1
        end = start + response_length

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
