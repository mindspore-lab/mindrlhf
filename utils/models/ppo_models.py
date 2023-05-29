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
import numpy as np
import mindspore
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import ReduceOp
from mindspore import ops, nn, Tensor, Parameter
from mindspore.common import dtype as mstype
from mindformers.models import GPT2LMHeadModel
from mindformers import AutoModel
from utils.generator import GeneratorMixin, ProcessLogits


class AdaptiveKLController(nn.Cell):
    """Adaptive KL Controller
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        super(AdaptiveKLController, self).__init__()
        self.value = Parameter(Tensor(init_kl_coef, mstype.float32), requires_grad=False)
        self.target = Tensor(target, mstype.float32)
        self.horizon = Tensor(horizon, mstype.int32)

    def construct(self, current, n_steps):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = ops.clip_by_value(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value = self.value * mult
        return self.value


class CausalLMHydraWithValueHead(nn.Cell, GeneratorMixin):
    """
    CausalLMHydraWithValueHead
    """

    def __init__(self, config, is_training=True):
        """
        Args:
            config: the configuration of GPT-2 model
            is_training (bool): `True` for train (finetune), `False` for evaluation.
        """
        super(CausalLMHydraWithValueHead, self).__init__()
        if not is_training:
            config.dropout_rate = 0.0
        self.config = config
        self.model = GPT2LMHeadModel(config)
        if config.from_pretrain:
            self.model = self.load_checkpoint_position_embedding(self.model, config)
        self.backbone = self.model.backbone
        self.lm_head = self.model.head
        self.e = Tensor(np.e)
        # self.lm_head = nn.Dense(config.hidden_size,
        #                        config.vocab_size,
        #                        weight_init=self.backbone.embedding.word_embedding.embedding_table,
        #                        has_bias=False).to_float(config.compute_dtype)
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.dtype = config.compute_dtype
        self.v_head0 = nn.Dense(config.hidden_size,
                                2*config.hidden_size,
                                weight_init=TruncatedNormal(0.02),
                                activation="relu",
                                has_bias=True).to_float(config.compute_dtype)

        self.v_head2 = nn.Dense(2*config.hidden_size,
                                1,
                                weight_init=TruncatedNormal(0.02),
                                has_bias=True).to_float(config.compute_dtype)

        self.squeeze = P.Squeeze(axis=-1)
        self.seq_length = self.config.seq_length
        self.pad_token_id = self.config.pad_token_id
        self.gather = P.GatherD()
        self.process_logits = ProcessLogits()
        self.top_k = Tensor(self.config.top_k, dtype=mstype.int32)
        self.eos_token_id = Tensor(self.config.eos_token_id, dtype=mstype.int32)
        self.max_decode_length = Tensor(self.config.max_decode_length, dtype=mstype.int32)

    def do_generate(self, input_ids, **x):
        return self.generate(input_ids, **x)

    def do_generate_graph(self, input_ids):
        return self.generate_graph(input_ids, self.top_k, self.eos_token_id, self.max_decode_length)
    
    def load_checkpoint_position_embedding(self, model, config):
        params_dict = mindspore.load_checkpoint(config.checkpoint_name_or_path)        
        slice_key = "backbone.embedding.position_embedding.embedding_table"
        params_dict_temp = {}
        for key, value in params_dict.items():
            if slice_key in key and params_dict[slice_key].shape[0] != config.seq_length:
                params_dict_temp["model." + slice_key] = Parameter(params_dict[slice_key][:config.seq_length, :])
                print("slice key:", slice_key, params_dict[slice_key].shape)
            else:
                params_dict_temp["model." + key] = value
        mindspore.load_param_into_net(model, params_dict_temp)
        print(f"{self.__class__.__name__} load ckpt successfully!")
        return model

    def construct(self,
                  input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                                 where 0 indicates padding position.

        Returns:
            lm_logits (Tensor): language model distribution with log_softmax, shape with[batch_size, seq_len, d_model].
        """

        output, embedding_table = self.backbone(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        batch_size, seq_length, d_model = self.shape(output)
        lm_logits = self.lm_head(output, embedding_table)
        # lm_logits = self.lm_head(output)
        lm_logits = P.Reshape()(lm_logits, (batch_size, seq_length, -1))
        value = self.v_head0(output)
        value = self.v_head2(value)
        value = self.squeeze(value)
        return lm_logits, value, output


class PPO_model(nn.Cell):
    """
    PPO_model
    """

    def __init__(self, config, is_training=True):
        """
        Args:
            config: the configuration of GPT-2 model
            is_training (bool): `True` for train (finetune), `False` for evaluation.
        """
        super(PPO_model, self).__init__()
        self.config = config

        self.policy_model = CausalLMHydraWithValueHead(config)
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

        self.cliprange_value = config.cliprange_value
        self.cliprange = config.cliprange
        self.vf_coef = config.vf_coef
        self.max = P.Maximum()
        self.cast = P.Cast()
        self.exp = P.Exp()
        self.stop_grad = P.StopGradient()
        self.gamma = 1
        self.lam = 0.95
        self.size = P.Size()
        self.sum_and_count = Tensor([[1, 1]])
        self.approx_kl = Parameter(0.0, requires_grad=False)
        self.kl_ctl = AdaptiveKLController(config.init_kl_coef, config.target, config.horizon)

    def construct(self, query_tensors, response_tensors, logprobs, values, rewards, attention_mask):
        """
        """
        old_logprobs = logprobs
        old_rewards = rewards
        old_values = values
        response_length = F.shape(old_rewards)[1]
        advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, response_length)
        tokens = response_tensors
        logits, values_pred, _ = self.policy_model(tokens, attention_mask)
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
        self.approx_kl = approx_kl
        return loss, logits, approx_kl

    def post_backward_callback(self, approx_kl, batch_size):
        return self.kl_ctl(approx_kl, n_steps=batch_size)

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


class PPOTrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(PPOTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.optimizer = optimizer
        self.grad_fn = mindspore.value_and_grad(network, None, self.optimizer.parameters, has_aux=True)

    def construct(self, query_tensors, sample_tensors, old_logprobs, old_values, old_rewards, label):
        (loss, _, approx_kl), grads = self.grad_fn(query_tensors, sample_tensors, old_logprobs, old_values, old_rewards, label)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss, approx_kl
