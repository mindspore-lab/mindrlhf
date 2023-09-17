import os
from mindspore import numpy as msnp
from mindspore import nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.common.dtype as mstype
# from mindtransformer.models.gpt import GPTModel
from mindformers.models import GPT2Model
from src.pangu_alpha import PanguAlpha_Model_No_Query_Layer
from mindspore.common.initializer import TruncatedNormal
# from mindspore.nn.transformer import TransformerOpParallelConfig
from mindformers.modules.transformer import TransformerOpParallelConfig
# from mindspore.nn.transformer.transformer import default_transformer_config
from mindformers.modules.transformer.transformer import default_transformer_config
from mindformers.models import T5ForConditionalGeneration, T5Tokenizer
from mindformers.modules.layers import LayerNorm, Dropout, Linear
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.communication.management as D
from mindspore import context
from typing import List
from tqdm import tqdm
import numpy as np
import mindspore
from mindspore.ops.operations._inner_ops import Send, Receive
try:
    from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig, AttentionMask

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)
from mindformers.models.bloom import BloomLMHeadModel, BloomConfig
from mindformers.models.pangualpha import PanguAlphaHeadModel, PanguAlphaConfig



class RewardModel(nn.Cell):
    def __init__(self, config):
        super(RewardModel, self).__init__()
        self.output_dtype = mstype.float16
        self.sequence_len = config.seq_length
        self.stopGrad = ops.stop_gradient
        self.cast = ops.Cast()
        self.shape = ops.Shape()
        self.squeeze = ops.Squeeze(axis=-1)
        self.reshape = P.Reshape()
        self.pad_token_id = config.pad_token_id
        if isinstance(config, PanguAlphaConfig):
            self.model_type = 'pangu'
        elif isinstance(config, BloomConfig):
            self.model_type = 'bloom'
        else:
            raise NotImplementedError("only support pangu and bloom")
        print("reward model model_type: ", self.model_type)

        if self.model_type == 'pangu':
            self.model = PanguAlphaHeadModel(config)
            self.backbone = self.model.backbone
        elif self.model_type == 'bloom':
            self.model = BloomLMHeadModel(config)
            self.backbone = self.model.transformer

        self.v_head0 = Linear(in_channels=config.hidden_size,
                              out_channels=4*config.hidden_size,
                              weight_init=TruncatedNormal(0.02),
                              activation="relu",
                              has_bias=True).to_float(mstype.float16)
        self.v_head2 = Linear(in_channels=4*config.hidden_size,
                              out_channels=1,
                              weight_init=TruncatedNormal(0.02),
                              has_bias=True).to_float(mstype.float16)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.v_head0.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_activation=((1, mp), ))
        self.v_head0.weight.parallel_optimizer = False
        self.v_head2.shard(strategy_matmul=((dp, mp), (1, mp)))
        self.v_head2.weight.parallel_optimizer = False
        self.v_head0.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.v_head2.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.sigmoid = nn.Sigmoid()
        self.sigmoid.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
        self.sub_shard = P.Sub().shard(((), (1, 1, 1)))

    def infer(self,
              input_ids,  # completion ids (after tokenized)
              end_indices,
              input_position=None, 
              attention_mask=None,
              init_reset=True, 
              batch_valid_length=None,  # Tensor[batch_size], max index of last token without padding, btw preferred and disfavored
              ):
        """
        """
        preferred_end_scores = []  # preferred completions' scores
        batch_size, seq_length = F.shape(input_ids)

        if self.model_type == 'pangu':
            if self.model.phase == "train":
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
                input_position = self.model.slice(input_position, (0, 0), (batch_size, seq_length), (1, 1))
            # [batch_size, seq_length, vocab_size]
            output_states, _ = self.backbone(tokens, input_position, attention_mask,
                                             init_reset, batch_valid_length)
        else:
            input_mask = self.model.not_equal(input_ids, self.model.eos_token_id).astype(mstype.float32)
            output_states, _ = self.backbone(input_ids, input_mask, init_reset, batch_valid_length)
 
        # batch_size = input_ids.shape[0] // 2  # reminder: there are two sets of data, for comparison
        #rewards = self.squeeze(self.vHead(backbone_output))

        rewards = self.v_head0(output_states)
        rewards = self.v_head2(rewards)
        rewards = self.reshape(rewards, (batch_size, seq_length))
        
        # sigmoid
        rewards  = self.sigmoid(rewards)
        rewards = rewards
       
        preferred_rewards = rewards  # [batch_size, seq_len]

        for i in range(batch_size):
            preferred_end_idx = end_indices[i].unsqueeze(0)
            preferred_end_scores.append(preferred_rewards[i][preferred_end_idx])

        preferred_end_scores = F.stack(preferred_end_scores, axis=0)
        return preferred_end_scores


class CriticModel(nn.Cell):
    def __init__(self, config):
        super(CriticModel, self).__init__()
        self.output_dtype = mstype.float16
        self.sequence_len = config.seq_length
        self.stopGrad = ops.stop_gradient
        self.cast = ops.Cast()
        self.shape = ops.Shape()
        self.squeeze = ops.Squeeze(axis=-1)
        self.reshape = P.Reshape()
        self.pad_token_id = config.pad_token_id
        if isinstance(config, PanguAlphaConfig):
            self.model_type = 'pangu'
        elif isinstance(config, BloomConfig):
            self.model_type = 'bloom'
        else:
            raise NotImplementedError("only support pangu and bloom")
        print("reward model model_type: ", self.model_type)

        if self.model_type == 'pangu':
            self.model = PanguAlphaHeadModel(config)
            self.backbone = self.model.backbone
        else:
            # self.model_type == 'bloom':
            self.model = BloomLMHeadModel(config)
            self.backbone = self.model.transformer

        self.v_head0 = Linear(in_channels=config.hidden_size,
                              out_channels=4*config.hidden_size,
                              weight_init=TruncatedNormal(0.02),
                              activation="relu",
                              has_bias=True).to_float(mstype.float16)
        self.v_head2 = Linear(in_channels=4*config.hidden_size,
                              out_channels=1,
                              weight_init=TruncatedNormal(0.02),
                              has_bias=True).to_float(mstype.float16)
        dp = config.parallel_config.data_parallel
        mp = config.parallel_config.model_parallel
        self.v_head0.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_activation=((1, mp), ))
        self.v_head0.weight.parallel_optimizer = False
        self.v_head2.shard(strategy_matmul=((dp, mp), (1, mp)))
        self.v_head2.weight.parallel_optimizer = False
        self.v_head0.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.v_head2.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.sigmoid = nn.Sigmoid()
        self.sigmoid.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.expand_dims = P.ExpandDims().shard(((dp, 1, 1),))
        self.sub_shard = P.Sub().shard(((), (1, 1, 1)))

    def construct(self, input_ids, attention_mask=None):
        batch_size, seq_length = F.shape(input_ids)
        if self.model_type == 'pangu':
            if self.model.phase == "train":
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
                input_position = self.model.slice(input_position, (0, 0), (batch_size, seq_length), (1, 1))
            # [batch_size, seq_length, vocab_size]
            output_states, _ = self.backbone(tokens, input_position, attention_mask,
                                             init_reset=None, batch_valid_length=None)
        else:
            input_mask = self.model.not_equal(input_ids, self.model.eos_token_id).astype(mstype.float32)
            output_states, _ = self.backbone(input_ids, input_mask, init_reset=None, batch_valid_length=None)
 
        values = self.v_head0(output_states)
        values = self.v_head2(values)
        values = self.reshape(values, (batch_size, seq_length))
        # sigmoid
        values = self.sigmoid(values)
        values = values - 0.5

        return values