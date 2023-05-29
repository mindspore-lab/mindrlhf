from mindspore import numpy as msnp
from mindspore import nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from src.pangu_alpha import PanguAlpha_Model_No_Query_Layer
from mindformers.modules.layers import LayerNorm, Linear
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore
try:
    from mindformers.modules.transformer import AttentionMask

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)


class RewardModel(nn.Cell):
    def __init__(self, config):
        super(RewardModel, self).__init__()
        self.output_dtype = mstype.float16
        self.sequence_len = config.seq_length
        self.stopGrad = ops.stop_gradient
        self.cast = ops.Cast()
        self.shape = ops.Shape()
        self.squeeze = ops.Squeeze(axis=-1)
        self.backbone = PanguAlpha_Model_No_Query_Layer(config)

        self.final_layernorm = LayerNorm((config.hidden_size,)).to_float(config.compute_dtype)
        if config.parallel_config.pipeline_stage > 1:
            self.final_layernorm.set_comm_fusion(2)
        else:
            self.final_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        self.final_layernorm.shard(((config.parallel_config.data_parallel, 1),))
        self.final_layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1
        in_features = config.hidden_size # change the parameter name to 'embedding_size'
        out_features = 1  # only a scalar is needed at the end
        # '''self.vHead = nn.Dense(in_features,
        #                       out_features,
        #                       weight_init=TruncatedNormal(0.02),
        #                       has_bias=False).to_float(mstype.float16)'''
        self.vHead = Linear(in_features, out_features, has_bias=False, transpose_b=True,
                            outer_batch=config.parallel_config.data_parallel,
                            param_init_type=config.param_init_type)
        self.vHead.shard(strategy_matmul=((config.parallel_config.data_parallel, 1),
                                          (1, 1)))
        self.sigmoid = nn.Sigmoid()

        self.vHead.pipeline_stage = config.parallel_config.pipeline_stage - 1
        self.sigmoid.pipeline_stage = config.parallel_config.pipeline_stage - 1

        self.get_attention_mask = AttentionMask(config.seq_length)
        self.print = ops.Print()

    def load_parameters(self, ckpt_file):
        param_dict = mindspore.load_checkpoint(ckpt_file)
        mindspore.load_param_into_net(self.backbone, param_dict)

    def infer(self,
              comp_ids,  # completion ids (after tokenized)
              comp_mask,  # completion attention masks (after tokenized),
              end_indices,  # Tensor[bs], max index of last token without padding, btw preferred and disfavored
              ):
        """
        infer.
 
        Args:
            comp_ids (Tensor): input completion sentences with shape [batch_size * 2, seq_len].
            comp_mask (Tensor): input completion sentences padding mask with shape [batch_size * 2, seq_len],
                                 where 0 indicates padding position.
 
        Returns:
            rewards (Tensor): rewards for each sentence, shape with[batch_size * 2].
        """
        preferred_end_scores = []  # preferred completions' scores
        bs, seq_length = F.shape(comp_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        attention_mask_pangu = self.get_attention_mask(comp_mask)
        backbone_output, _ = self.backbone(comp_ids, 
                                           input_position, 
                                           attention_mask_pangu, 
                                           True, 
                                           None)
        backbone_output = self.final_layernorm(backbone_output)

        backbone_output = self.cast(backbone_output, self.output_dtype)
        rewards = self.vHead(backbone_output)
        rewards = rewards.reshape(comp_ids.shape[0], seq_length)
        
        preferred_rewards = rewards  # [bs, seq_len]
        for i in range(bs):
            preferred_end_idx = end_indices[i].unsqueeze(0)
            preferred_end_scores.append(preferred_rewards[i][preferred_end_idx])

        preferred_end_scores = F.stack(preferred_end_scores, axis=0)
        return preferred_end_scores
        
    def construct(self,
                  comp_ids,  # completion ids (after tokenized)
                  comp_mask,  # completion attention masks (after tokenized),
                  end_indices,  # Tensor[bs], max index of last token without padding, btw preferred and disfavored
                  truncate_ranges,
                  inference: bool
                  ):
        """
        Construct network.

        Args:
            comp_ids (Tensor): input completion sentences with shape [batch_size * 2, seq_len].
            comp_mask (Tensor): input completion sentences padding mask with shape [batch_size * 2, seq_len],
                                 where 0 indicates padding position.

        Returns:
            rewards (Tensor): rewards for each sentence, shape with[batch_size * 2].
        """
        preferred_end_scores = [Tensor(-1)]  # preferred completions' scores
        disfavored_end_scores = [Tensor(-1)]  # disfavored completions' scores
        loss = 0

        # backbone_output, present_layer, embedding_table = self.backbone(comp_ids, comp_mask)
        bs, seq_length = F.shape(comp_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (bs, 1))
        attention_mask_pangu = self.get_attention_mask(comp_mask)
        backbone_output, _ = self.backbone(comp_ids, 
                                                         input_position, 
                                                         attention_mask_pangu, 
                                                         True, 
                                                         None)
        backbone_output = self.final_layernorm(backbone_output)
        backbone_output = self.cast(backbone_output, self.output_dtype)
        _, seq_length, _ = self.shape(backbone_output)

        bs = comp_ids.shape[0] // 2
        rewards = self.squeeze(self.vHead(backbone_output))
        preferred_rewards = rewards[:bs]
        disfavored_rewards = rewards[bs:]

        if inference:
            for i in range(bs):
                preferred_end_idx = end_indices[i]
                preferred_end_scores.append(preferred_rewards[i, preferred_end_idx])
        else:
            for i in range(bs):
                diff_rewards = preferred_rewards[i] - disfavored_rewards[i]
                end_idx = end_indices[i].astype('int32')
                input_indices = Tensor(truncate_ranges[i])
                truncated = ops.gather(input_params=diff_rewards, input_indices=input_indices, axis=0)
                preferred_end_scores.append(preferred_rewards[i][end_idx])
                disfavored_end_scores.append(disfavored_rewards[i][end_idx])
                output = self.sigmoid(truncated).astype('float16')
                output_log = msnp.log(output)
                output_log_mean = ops.mean(output_log, axis=0)
                loss += (-1) * output_log_mean
            loss /= bs

        return loss, preferred_end_scores[1:], disfavored_end_scores[1:]
