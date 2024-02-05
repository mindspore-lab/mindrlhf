# Copyright 2023 Huawei Technologies Co., Ltd
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
"""baichuan2 reward model"""
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../../")))
# print('======path:', os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../../../")))
import copy
import mindspore
from mindspore import nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindformers.models import BaseModel
from mindformers.modules.layers import Linear
from mindformers.tools.register.register import MindFormerModuleType, MindFormerRegister
from mindspore import ops as P
from mindspore.ops import functional as F
from mindformers.models.llama import LlamaForCausalLM, LlamaModel, LlamaConfig
from mindformers.core.loss import CompareLoss
from mindformers.tools.logger import logger
from mindrlhf.models.baichuan2.baichuan2_7b import Baichuan7BV2ForCausalLM, Baichuan7BV2Model

__all__ = ['VHead', 'Baichuan7BReward']

class VHead(nn.Cell):
    r"""Head for Llama to get the logits of each token in the vocab."""
    def __init__(self, config=None):
        super().__init__()
        dp = config.parallel_config.data_parallel
        mp = 1
        self.v_head0 = Linear(in_channels=config.hidden_size,
                            out_channels=1,
                            has_bias=False).to_float(mstype.float16)
        self.v_head0.shard(strategy_matmul=((dp, mp), (mp, 1)))
        self.v_head0.pipeline_stage = config.parallel_config.pipeline_stage - 1

    def construct(self, output_states):
        """
        construct function for vhead
        """
        return self.v_head0(output_states)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Baichuan7BReward(BaseModel):
    r"""
        Provide llama reward model training loss or logits through network.
        Args:
            config (LlamaConfig): The config of LlamaModel.

        Returns:
            Tensor, the loss or logits of the network.
        """

    def __init__(self, config=None):
        config = config if config is not None else LlamaConfig()
        super(Baichuan7BReward, self).__init__(config, auto_prefix=False)
        mp = config.parallel_config.model_parallel
        self.seq_length = config.seq_length
        parallel_config = config.parallel_config
        # self.transformer = Baichuan7BV2ForCausalLM(config).model
        self.transformer = Baichuan7BV2Model(config)
        self.v_head0 = VHead(config)
        if parallel_config.pipeline_stage > 1:
            self.v_head0.pipeline_stage = parallel_config.pipeline_stage - 1
            self.transformer.embedding.word_embedding.embedding_table.add_pipeline_stage(self.v_head0.pipeline_stage)

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Bloom Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Bloom Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1

        checkpoint_path = config.checkpoint_path
        if checkpoint_path:
            param_dict = mindspore.load_checkpoint(checkpoint_path)
            self.transformer.update_parameters_name()
            keys = self.transformer.parameters_dict()
            new_param_dict = {}
            for k, v in param_dict.items():
                if "lm_head" in k:
                    continue
                if k not in keys:
                    k = k.replace("model.", "")
                    new_param_dict[k] = v
                else:
                    new_param_dict[k] = v
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.transformer, new_param_dict)
            print("param_not_load:", param_not_load)
        self.loss = CompareLoss(config=loss_parallel_config)
        self.load_checkpoint(config)
        self.gatherd = P.GatherD()
        self.slice = P.StridedSlice().shard(((1, 1),))
        self.slice_ind = P.StridedSlice().shard(((1,),))

    def construct(self,
                  input_ids,
                  position_id=None,
                  attention_mask=None,
                  loss_mask=None,
                  end_ind=None):
        """
        construct function for reward model
        """
        output_states = self.transformer(input_ids)

        # [bs, seq, hidden_size]
        logits = self.v_head0(output_states)
        # [bs, seq, 1]
        logits = logits.squeeze(-1)
         # [bs, seq]
        logits = F.reshape(logits, (-1, self.seq_length))
        loss, chosen_end_scores, reject_end_scores = self.loss(logits, loss_mask, end_ind)
        return loss

    def eval(self,
             input_ids,
             end_ind):
        batch_size, seq_length = F.shape(input_ids)
        batch_size = batch_size // 2
        output_states = self.transformer(input_ids)
        logits = self.v_head0(output_states)
        logits = logits.squeeze(-1)
        rewards = F.reshape(logits, (-1, seq_length))
        chosen_rewards = self.slice(rewards, (0, 0), (batch_size, seq_length), (1, 1))
        rejected_rewards = self.slice(rewards, (batch_size, 0), (2 * batch_size, seq_length), (1, 1))
        end_ind_chosen = self.slice_ind(end_ind, (0,), (batch_size,), (1,))
        end_ind_reject = self.slice_ind(end_ind, (batch_size,), (2 * batch_size,), (1,))
        end_inds = P.Concat()((end_ind_chosen, end_ind_reject))
        end_inds = end_inds.reshape((2, -1))
        end_ind_final, _ = P.max(end_inds, axis=0)
        end_ind_final = end_ind_final.reshape((-1, 1))
        chosen_end_scores = self.gatherd(chosen_rewards, 1, end_ind_final-1)
        reject_end_scores = self.gatherd(rejected_rewards, 1, end_ind_final-1)
        return chosen_end_scores, reject_end_scores

    def infer(self,
              input_ids,
              end_ind):
        batch_size, seq_length = F.shape(input_ids)
        output_states = self.transformer(input_ids)
        logits = self.v_head0(output_states)
        logits = logits.squeeze(-1)
        rewards = F.reshape(logits, (-1, seq_length))
        end_ind = end_ind.reshape((-1, 1))
        scores = self.gatherd(rewards, 1, end_ind-1)
        return scores