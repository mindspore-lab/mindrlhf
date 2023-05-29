import numpy as np
from utils.models.reward_model_pangu import RewardModel
import mindspore
from mindspore import Tensor, ops, mutable
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.communication.management as D
import mindspore.nn as nn
#from trlx.utils import Clock
from mindspore.dataset import GeneratorDataset
from dataclasses import dataclass
from utils.models.ppo_models_pangu import PanguWithValueHead, PPO_model, SRNet, SRNet_1, SRNet_2
from utils.configs import PPOConfig
import mindspore.common.dtype as mstype
from mindspore import numpy as msnp
from mindspore import context
from mindspore.dataset import MindDataset
from utils.utils import set_pipeline_parallel_context
from ppo_trainer import PPOElement


class RewardScores(nn.Cell):
    def __init__(self, model_config):
        super(RewardScores, self).__init__()
        self.pad_id = model_config.pad_token
        self.reward_model = RewardModel(model_config)

    def get_scores(self, samples):
        attn_masks = (samples != self.pad_id).to(mstype.float32)
        end_indices = (attn_masks.sum(axis=1) - 1).to(mstype.int32)
        bs_scores = self.reward_model.infer(samples, attn_masks, end_indices)    
        return bs_scores

    def construct(self, samples, original_samples):
        original_scores = self.get_scores(original_samples)
        scores = self.get_scores(samples)
        norms_scores = scores - original_scores
        return scores

    def load(self, path=None):
        param_dict = mindspore.load_checkpoint(path)
        param_not_load = mindspore.load_param_into_net(self.reward_model, param_dict)


class LogprobsOfLabels(nn.Cell):
    def __init__(self):
        super(LogprobsOfLabels, self).__init__()
        self.cast = ops.Cast()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gatherd = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)
    def construct(self, logits, labels):
        """Compute log probs of the labels"""
        labels = self.cast(labels, mindspore.int32)
        logprobs = self.log_softmax(logits)
        logprobs_labels = self.gatherd(logprobs, -1, self.unsqueeze(labels, -1))
        return self.squeeze(logprobs_labels)



class PanguPPOTrainer:
    def __init__(self, ppo_config=None, sft_model_config=None, rm_model_config=None, opt=None, 
                 prompt_dataset=None, val_dataset=None):
        # self.prompt_dataloader = GeneratorDataset(generator, column_names=["input_ids", "attention_mask"])
        # self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=10)
        # self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
        # load data from mindrecord file
        # 可以注释掉
        self.mind_dataset_dir = ["../../TLDR_data/train/tldr_train_prompts.mindrecord"]
        columns_to_project = ["prompt_ids", "prompt_mask", "original_sample_ids", "original_sample_mask"]
        self.prompt_dataloader = MindDataset(self.mind_dataset_dir).project(columns=columns_to_project)
        self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=ppo_config.chunk_size)
        self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
        self.val_iterator = val_dataset

        self.ppo_config = ppo_config
        self.sft_model_config = sft_model_config
        self.rm_model_config = rm_model_config
        self.opt = opt

        # 可以传入
        set_pipeline_parallel_context(parallel_mode=self.opt.parallel_mode, full_batch=self.opt.full_batch,
                    optimizer_shard=self.opt.optimizer_shard, stage_num=self.sft_model_config.parallel_config.pipeline_stage, 
                    enable_alltoall=self.opt.enable_alltoall)

        self.policy_model = PanguWithValueHead(sft_model_config, self.ppo_config)
        self.ppo_model = PPO_model(ppo_config, self.policy_model, self.opt)
        # mindspore.load_checkpoint("./ms_ppo.ckpt", self.ppo_model.policy_model) 

        self.ref_model = PanguWithValueHead(sft_model_config, ppo_config)
        # mindspore.load_checkpoint("./ms_ppo.ckpt", self.ref_model) 
        self.ref_model.set_train(False)

        self.ref_mean = 0
        self.ref_std = 0
        self.cliprange_reward = 10.0
        self.ppo_elements = []
        
        set_pipeline_parallel_context(parallel_mode=self.opt.parallel_mode, full_batch=self.opt.full_batch,
                    optimizer_shard=self.opt.optimizer_shard, stage_num=self.rm_model_config.parallel_config.pipeline_stage, 
                    enable_alltoall=self.opt.enable_alltoall)
        self.reward_fn = RewardScores(rm_model_config)
        self.reward_fn.set_train(False)
 
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)

        device_num = D.get_group_size()
        rank_id = D.get_rank()
        sft_stage_num = self.sft_model_config.parallel_config.pipeline_stage
        sft_per_stage_num = int(device_num / sft_stage_num)
        sft_stage_id = int(rank_id / sft_per_stage_num)
        rm_stage_num = self.rm_model_config.parallel_config.pipeline_stage
        rm_per_stage_num = int(device_num / rm_stage_num)
        rm_stage_id = int(rank_id / rm_per_stage_num)
        bsz = self.ppo_config.chunk_size
        print("bsz: ", bsz)
        seq_length = self.ppo_config.seq_length
        vocab_size = self.policy_model.model_config.vocab_size

        self.sr_net_1 = SRNet_1(sft_stage_id, rank_id, sft_stage_num, device_num, bsz, seq_length, vocab_size)
        self.sr_net_2 = SRNet_2(rm_stage_id, rank_id, rm_stage_num, device_num, bsz, seq_length, vocab_size)
        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=1)
        self.gather = ops.Gather(-1)
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gatherd = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)
        self.tile = P.Tile()
        self.logprobs_of_labels = LogprobsOfLabels()

    
    '''def logprobs_of_labels(self, logits, labels):
        """Compute log probs of the labels"""
        labels = self.cast(labels, mindspore.int32)
        logprobs = self.log_softmax(logits)
        logprobs_labels = self.gatherd(logprobs, -1, self.unsqueeze(labels, -1))
        return self.squeeze(logprobs_labels)'''

    def gather_ppo_elements(self, data):
        self.ppo_elements = data

    def generate(self, input_ids, attn_masks=None):
        """generate response"""
        input_ids_list = input_ids.asnumpy().tolist()
        outputs = self.ppo_model.generate(input_ids_list)

        prompt_len = (np.array(input_ids_list) != self.ppo_config.pad_token_id).astype(int).sum(1)
        response_len = [(outputs[i] != self.ppo_config.pad_token_id).astype(int).sum() for i in range(len(outputs))]
        
        left_padding_prompt = np.ones((len(input_ids_list), self.ppo_config.max_prompt_length)) * self.ppo_config.pad_token_id
        outputs_array = np.ones((len(input_ids_list), self.ppo_config.seq_length)) * self.ppo_config.pad_token_id
        resposne_array = np.ones((len(input_ids_list), self.ppo_config.max_decode_length)) * self.ppo_config.pad_token_id
        for i in range(len(input_ids_list)):
            left_padding_prompt[i, self.ppo_config.max_prompt_length - prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]
            outputs_array[i, self.ppo_config.max_prompt_length - prompt_len[i]:
                             self.ppo_config.max_prompt_length - prompt_len[i] + response_len[i] + 1] = outputs[i]
            resposne_array[i, :] = outputs_array[i, self.ppo_config.max_prompt_length: ]

        left_padding_prompt = left_padding_prompt.astype(int).tolist()
        output_ids_list = outputs_array.astype(int).tolist()
        response_ids_list = resposne_array.astype(int).tolist()
        return left_padding_prompt, output_ids_list, response_ids_list
    
    def partition(self, prompt_tensors, samples):
        n_samples: int = samples.shape[0]
        response_tensors = []
        for ix in range(n_samples):
            start = np.max(np.nonzero(np.not_equal(prompt_tensors[ix], self.ppo_config.pad_token_id))) + 1
            response_tensors.append(samples[ix, start: int(start + self.ppo_config.max_decode_length)])
        return response_tensors

    def generate_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        self.policy_model.set_train(False)
        self.ref_model.set_train(False)
        ppo_rl_elements = []
        while len(ppo_rl_elements) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
                batch = next(self.prompt_iterator)

            prompt_tensors = self.cast(batch[0], mstype.int32) 
            prompt_sizes = [prompt_tensors.shape[1]] * len(prompt_tensors)
            prompt_tensor, samples, responses = self.generate(prompt_tensors)
            samples = Tensor(samples)
            prompt_tensor = Tensor(prompt_tensor)

            set_pipeline_parallel_context(parallel_mode=self.opt.parallel_mode, full_batch=self.opt.full_batch,
                    optimizer_shard=self.opt.optimizer_shard, stage_num=self.rm_model_config.parallel_config.pipeline_stage, 
                    enable_alltoall=self.opt.enable_alltoall)
            samples = samples.to(mstype.int32)
            original_samples = batch[2].to(mstype.int32)
            scores = self.reward_fn(samples, original_samples=original_samples)
            context.reset_auto_parallel_context()
            scores = self.sr_net_2(scores)

            print("scores: ", scores.shape)
            set_pipeline_parallel_context(parallel_mode=self.opt.parallel_mode, full_batch=self.opt.full_batch,
                    optimizer_shard=self.opt.optimizer_shard, stage_num=self.sft_model_config.parallel_config.pipeline_stage, 
                    enable_alltoall=self.opt.enable_alltoall)

            attention_mask = ops.not_equal(samples, PPOConfig.pad_token_id)
            attention_mask = self.cast(attention_mask, mstype.float32)
            attention_mask_pangu = self.ppo_model.get_attention_mask(attention_mask)
            samples = self.cast(samples, mstype.int32)

            input_mask = F.cast(F.not_equal(samples, PPOConfig.pad_token_id), mstype.float32)
            bs, seq_length = F.shape(samples)
            input_position = F.tuple_to_array(F.make_range(seq_length))
            input_position = self.tile(input_position, (bs, 1))
            
            ppo_out = self.ppo_model.policy_model(samples, input_position, attention_mask_pangu)

            ref_out = self.ref_model(samples, input_position, attention_mask_pangu)
            context.reset_auto_parallel_context()
            logits = self.ppo_model.sr_net(mutable(ppo_out))
            values = self.sr_net_1(mutable(ppo_out))
            ref_logits = self.ppo_model.sr_net(mutable(ref_out))

            logprobs = self.logprobs_of_labels(logits[:, :-1, :], samples[:, 1:])
            ref_logprobs = self.logprobs_of_labels(ref_logits[:, :-1, :], samples[:, 1:])
            values = values.asnumpy()
            logprobs = logprobs.asnumpy()
            ref_logprobs = ref_logprobs.asnumpy()
            values = values[:, :-1]

            n_samples: int = samples.shape[0]
            starts = (np.ones(n_samples) * (self.ppo_config.max_prompt_length - 1)).astype(int)
            ends = (np.ones(n_samples) * (self.ppo_config.seq_length - 1)).astype(int)
            
            valid_length_response = [(np.array(responses[i]) != self.ppo_config.pad_token_id).astype(int).sum()
                                     for i in range(n_samples)]
            all_values = []
            all_logprobs = []
            for ix in range(n_samples):
                all_values.append(values[ix, starts[ix]:ends[ix]])
                all_logprobs.append(logprobs[ix, starts[ix]:ends[ix]])

            kl_divergence_estimate = self.ppo_model.kl_ctl.value.asnumpy() * (logprobs - ref_logprobs) 
            kl_divergence_estimate = [rs[starts[ix]:ends[ix]] for ix, rs in enumerate(kl_divergence_estimate)]
            rollout_count = 0
            for sample_idx in range(n_samples):
                sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]
                rewards = sample_kl_divergence_estimate 
                rewards[int(valid_length_response[sample_idx] - 1)] += scores[sample_idx] 
                
                all_logprobs[sample_idx][int(valid_length_response[sample_idx]): ] = 0.0
                all_values[sample_idx][int(valid_length_response[sample_idx]): ] = 0.0
                all_values = np.array(all_values).squeeze()
                rewards[int(valid_length_response[sample_idx]): ] = 0.0

                ppo_rl_elements.append(
                    PPOElement(
                        query_tensor=prompt_tensors[sample_idx].asnumpy(),
                        sample_tensor=samples[sample_idx].asnumpy(),
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )
                rollout_count += 1
        self.gather_ppo_elements(ppo_rl_elements)
