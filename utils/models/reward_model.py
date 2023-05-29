import os
import mindspore
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import numpy as msnp
from mindspore import nn, Tensor
from mindformers.models import GPT2Model
from mindspore.common.initializer import TruncatedNormal


class RewardModel(nn.Cell):
    def __init__(self, config):
        super(RewardModel, self).__init__()
        self.output_dtype = mstype.float16
        self.sequence_len = config.seq_length
        self.stopGrad = ops.stop_gradient
        self.cast = ops.Cast()
        self.shape = ops.Shape()
        self.squeeze = ops.Squeeze(axis=-1)
        # Load the pre-trained reward model, config and Tokenizer
        self.backbone = GPT2Model(config)
        self.vocab_size = config.vocab_size
        self.vHead = nn.Dense(in_channels=config.embedding_size,
                              out_channels=1,
                              weight_init=TruncatedNormal(0.02),
                              has_bias=False).to_float(mstype.float16)
        self.sigmoid = nn.Sigmoid()
        self.gatherd = ops.GatherD()
        self.unsqueeze = ops.ExpandDims()
        self.load_checkpoint(config)

    def load_checkpoint(self, config):
        if os.path.exists(config.checkpoint_name_or_path):
            param = mindspore.load_checkpoint(config.checkpoint_name_or_path)
            mindspore.load_param_into_net(self, param)
            print(f"{self.__class__.__name__} load ckpt successfully!")

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
        preferred_end_scores = []
        backbone_output, _ = self.backbone(comp_ids, comp_mask)
        backbone_output = self.cast(backbone_output, self.output_dtype)
        bs = comp_ids.shape[0] // 2  # reminder: there are two sets of data, for comparison
        rewards = self.squeeze(self.vHead(backbone_output))
        preferred_rewards = rewards[:bs]  # [bs, seq_len]
        for i in range(bs):
            preferred_end_idx = end_indices[i]
            preferred_end_scores.append(preferred_rewards[i, preferred_end_idx])
        return preferred_end_scores
        
    def construct(self,
                  comp_ids,  # completion ids (after tokenized)
                  comp_mask,  # completion attention masks (after tokenized),
                  end_indices,  # Tensor[bs], max index of last token without padding, btw preferred and disfavored
                  truncate_ranges=None,
                  inference=False
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
        preferred_end_scores = []  # preferred completions' scores
        disfavored_end_scores = []  # disfavored completions' scores
        loss = 0
        backbone_output, _ = self.backbone(comp_ids, comp_mask)
        backbone_output = self.cast(backbone_output, self.output_dtype)
        bs = comp_ids.shape[0] // 2  # reminder: there are two sets of data, for comparison
        rewards = self.squeeze(self.vHead(backbone_output))
        preferred_rewards = rewards[:bs]  # [bs, seq_len]
        disfavored_rewards = rewards[bs:]

        if inference:
            for i in range(bs):
                preferred_end_idx = end_indices[i]
                preferred_end_scores.append(preferred_rewards[i, preferred_end_idx])
            return preferred_end_scores
        else:
            # Index into the correct rewards.
            for i in range(bs):
                # TRAINING case. Check if there is any padding otherwise take length of sequence
                diff_rewards = preferred_rewards[i] - disfavored_rewards[i]
                end_idx = end_indices[i].astype('Int64')
                input_indices = Tensor(truncate_ranges[i])
                truncated = ops.gather(input_params=diff_rewards, input_indices=input_indices, axis=0)
                preferred_end_scores.append(preferred_rewards[i][end_idx])
                disfavored_end_scores.append(disfavored_rewards[i][end_idx])
                output = self.sigmoid(truncated).astype('float16')
                output_log = msnp.log(output)
                output_log_mean = ops.mean(output_log, axis=0)
                loss += (-1) * output_log_mean
            loss /= bs

        return loss, preferred_end_scores, disfavored_end_scores
