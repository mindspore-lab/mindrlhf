import time
from dataclasses import dataclass
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindrlhf.configs.ppo_configs import PPOConfig
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.dataset import MindDataset

from mindrlhf.models.reward_model import RewardModel, CriticModel
from mindrlhf.models.ppo_models import CausalLMHydraWithValueHead, PPO_model

def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params

@dataclass
class PPOData:
    query_tensor: Tensor
    response_tensor: Tensor
    logprobs: Tensor
    values: Tensor
    rewards: Tensor
    advantages: Tensor
    returns: Tensor
    pretrain_ids: Tensor
    loss_mask: Tensor
    attention_mask: Tensor

def get_first_diverge_indices(preferred_comp_ids,  # shape = batch_size * seq_length
                              disfavored_comp_ids  # shape = batch_size * seq_length
                              ):
    is_equal = Tensor(preferred_comp_ids == disfavored_comp_ids).astype('float32')
    print("is_equal is: ", is_equal)
    first_diverge_indices = is_equal.sum(axis=1, dtype=mindspore.int32)
    return first_diverge_indices

class RewardFn(nn.Cell):
    def __init__(self, model_config):
        super(RewardFn, self).__init__()

        self.ckpt_path = model_config.checkpoint_name_or_path
        model_config.checkpoint_name_or_path = ""
        self.pad_token = model_config.pad_token_id
        self.reward_model = RewardModel(model_config)
        self.not_equal = P.NotEqual()

        if self.ckpt_path:
            param_dict = mindspore.load_checkpoint(self.ckpt_path)
            print("Begin to load reward model ckpt from: ", self.ckpt_path, flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.reward_model, param_dict)
            print("Parameter not loaded: ", param_not_load, flush=True)
            print("Ckpt not loaded: ", ckpt_not_load, flush=True)

    def get_scores(self, samples):
        attn_masks = self.not_equal(samples, self.pad_token).astype(mstype.float32)
        end_indices = (attn_masks.sum(axis=1) - 1).to(mstype.int32)
        bs_scores = self.reward_model.infer(samples, end_indices)    
        return bs_scores

    def construct(self, samples, original_sample=None):
        scores = self.get_scores(samples)
        if original_sample is not None:
            ori_scores = self.get_scores(original_sample)
        else:
            ori_scores = 0.1
        return scores - ori_scores

class PPOTrainer:
    def __init__(self, 
                 ppo_config=None, 
                 sft_model_config=None, 
                 ref_model_config=None, 
                 critic_model_config=None,
                 rm_model_config=None):
        self.ppo_config = ppo_config
        self.mind_dataset_dir = ppo_config.mind_dataset_dir
        columns_to_project = ["prompt_ids", "pretrain_ids", "loss_mask"]
        mindspore.dataset.config.set_seed(2023)
        print("mind_dataset_dir", self.mind_dataset_dir)
        dataset = MindDataset(self.mind_dataset_dir).project(columns=columns_to_project)
        self.prompt_dataloader = dataset.take(ppo_config.num_rollouts)
        self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=ppo_config.chunk_size \
                                                              * sft_model_config.parallel_config.data_parallel)
        self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()

        self.sft_model_config = sft_model_config
        self.rm_model_config = rm_model_config
        policy_model = CausalLMHydraWithValueHead(sft_model_config, self.ppo_config)
        if sft_model_config.checkpoint_name_or_path:
            param_dict = mindspore.load_checkpoint(sft_model_config.checkpoint_name_or_path)
            new_param_dict = {k.replace("transformer", "backbone").replace("backbone.backbone", "backbone.transformer"): v for k, v in param_dict.items()}
            print(f"=====begin to load policy model from: {sft_model_config.checkpoint_name_or_path}", flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(policy_model, new_param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)
        # param_dict = mindspore.load_checkpoint("./sft_model.ckpt")
        # param_not_load, ckpt_not_load = mindspore.load_param_into_net(policy_model, param_dict)
        # print(f"param not load: {param_not_load}", flush=True)
        # print(f"ckpt not load: {ckpt_not_load}", flush=True)
        
        critic_model = CriticModel(critic_model_config)
        if critic_model_config.checkpoint_name_or_path:
            param_dict = mindspore.load_checkpoint(critic_model_config.checkpoint_name_or_path)
            new_param_dict = {k.replace("reward_model.model.", "").replace("transformer", "backbone").replace("backbone.backbone", "backbone.transformer"): v for k, v in param_dict.items()}
            print(f"=====begin to load critic model from: {critic_model_config.checkpoint_name_or_path}", flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(critic_model, new_param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)
        # param_dict = mindspore.load_checkpoint("critic_model.ckpt")
        # param_not_load, ckpt_not_load = mindspore.load_param_into_net(critic_model, param_dict)
        # print(f"param not load: {param_not_load}", flush=True)
        # print(f"ckpt not load: {ckpt_not_load}", flush=True)
        self.ppo_model = PPO_model(ppo_config, policy_model, critic_model)

        self.ref_model = CausalLMHydraWithValueHead(ref_model_config, self.ppo_config)
        if ref_model_config.checkpoint_name_or_path:
            param_dict = mindspore.load_checkpoint(ref_model_config.checkpoint_name_or_path)
            new_param_dict = {k.replace("transformer", "").replace("transformer", "backbone").replace("backbone.backbone", "backbone.transformer"): v for k, v in param_dict.items()}
            print(f"=====begin to load critic model from: {ref_model_config.checkpoint_name_or_path}", flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.ref_model, new_param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)
        self.ref_model.model.set_train(False)
        # param_dict = mindspore.load_checkpoint("./sft_model.ckpt")
        # param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.ref_model, param_dict)
        # print(f"param not load: {param_not_load}", flush=True)
        # print(f"ckpt not load: {ckpt_not_load}", flush=True)

        self.ref_mean = 0
        self.ref_std = 0
        self.cliprange_reward = 10.0
        self.store = []

        self.reward_fn = RewardFn(rm_model_config)
        self.reward_fn.set_train(False)
        self.reward_fn.reward_model.set_train(False)
        self.reward_fn.reward_model.model.set_train(False)
        # param_dict = mindspore.load_checkpoint("reward_model.ckpt")
        # param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.reward_fn.reward_model, param_dict)
        # print(f"param not load: {param_not_load}", flush=True)
        # print(f"ckpt not load: {ckpt_not_load}", flush=True)
 
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)
        self.depend = P.Depend()
        
        # # save sft, reward, critic model ckpt
        # mindspore.save_checkpoint(policy_model, "sft_model.ckpt")
        # mindspore.save_checkpoint(critic_model, "critic_model.ckpt")
        # mindspore.save_checkpoint(self.reward_fn.reward_model, "reward_model.ckpt")
        

    def push_to_store(self, data):
        self.store = data

    def generate(self, input_ids, attn_masks=None):
        print("input_ids", input_ids.shape)
        input_ids_list = input_ids.asnumpy().tolist()
        
        prompt_len = (np.array(input_ids_list) != self.ppo_config.pad_token_id).astype(int).sum(1)
        left_padding_prompt = np.ones((len(input_ids_list), self.ppo_config.max_prompt_length)) * self.ppo_config.pad_token_id
        resposne_array = np.ones((len(input_ids_list), self.ppo_config.max_decode_length)) * self.ppo_config.pad_token_id
        samples = np.ones((len(input_ids_list), self.ppo_config.seq_length)) * self.ppo_config.pad_token_id

        generate_begin_time = time.time()
        # outputs = self.ppo_model.generate(input_ids_list)
        print("input_ids shape", input_ids.shape)
        outputs = self.ppo_model.policy_model.model.generate(input_ids_list, max_length=self.ppo_config.seq_length)
        print("Generating elapsed time: ", time.time() - generate_begin_time)
        # exit()
        for i in range(len(input_ids_list)):
            x = outputs[i][prompt_len[i]: prompt_len[i] + self.ppo_config.max_decode_length]
            resposne_array[i, :len(x)] = x
            print(resposne_array)
            p = outputs[i]
            samples[i, :len(p)] = p
            left_padding_prompt[i, self.ppo_config.max_prompt_length - prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]
        return Tensor(samples, mstype.int32), Tensor(resposne_array, mstype.int32), Tensor(left_padding_prompt, mstype.int32)

    
    def partition(self, prompt_tensors, samples):
        n_samples: int = samples.shape[0]
        response_tensors = []
        for ix in range(n_samples):
            # get the start_idx of the response in `prompt_tensors`,
            # where `prompt_tensors` is the concatenated prompt and response
            start = np.max(np.nonzero(np.not_equal(prompt_tensors[ix], self.ppo_config.pad_token_id))) + 1
            response_tensors.append(samples[ix, start: int(start + self.ppo_config.max_decode_length)])
        return response_tensors

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        self.ppo_model.policy_model.model.set_train(False)
        self.ppo_model.critic_model.model.set_train(False)
        self.ref_model.model.set_train(False)
        self.reward_fn.reward_model.set_train(False)
        ppo_rl_elements = []
        scores_record = []
        while len(ppo_rl_elements) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
                print(batch)
            except StopIteration:
                mindspore.dataset.config.set_seed(2023)
                self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
                batch = next(self.prompt_iterator)

            # batch[0]: prompt, right padding to max_prompt_length
            # batch[1]: concatenated pretrain data, right padding to max length
            # batch[2]: loss mask of pretrain data
            prompt_tensors = Tensor(batch[0], mstype.int32)
            pretrain_ids = Tensor(batch[1], mstype.int32)
            # loss_mask = Tensor(batch[2], mstype.float32)
            loss_mask = batch[2][:,1:]

            start_time = time.time()
            print("generation start at {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            self.ppo_model.policy_model.model.add_flags_recursive(use_past=self.ppo_config.use_past)
            samples, resposne_array, left_padding_prompt = self.generate(prompt_tensors)
            end_time = time.time()
            print("generate end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

            # samples: prompt + generated response, right padding to seq_length
            samples = samples.asnumpy()
            resposne_array = resposne_array.asnumpy()
            left_padding_prompt = left_padding_prompt.asnumpy()
            self.ppo_model.policy_model.model.add_flags_recursive(use_past=False)

            # samples: prompt + generated response, right padding to seq_length
            samples = Tensor(samples, mstype.int32)

            start_time = time.time()
            print("reward model start at {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            scores = self.reward_fn(samples)
            end_time = time.time()
            print("reward model end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)
            print("scores: \n", scores, flush=True)
            scores_record += scores.asnumpy().tolist()
            self.ppo_model.policy_model.model.set_train(False)
            self.ref_model.model.set_train(False)

            # all_tokens: [pad, ..., pad, `prompt`, `response`, pad, ..., pad]
            all_tokens = np.concatenate((left_padding_prompt, resposne_array), axis=1)
            all_tokens = Tensor(all_tokens, mstype.int32)

            start_time = time.time()
            print(f"all_tokens shape:", all_tokens.shape)
            print("policy model start at {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            logprobs = self.ppo_model.policy_model(all_tokens, samples=all_tokens)
            # logprobs, values = self.ppo_model.policy_model(all_tokens, samples=all_tokens, return_value=True)
            print(f"log_probs:", logprobs, logprobs.shape)
            # logprobs = self.ppo_model.policy_model(all_tokens)
            # logprobs = logprobs.reshape((-1, self.ppo_config.seq_length))
            end_time = time.time()
            print("policy model end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

            start_time = time.time()
            print("critic model start at {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            values = self.ppo_model.critic_model(all_tokens)
            # values = values.reshape((-1, self.ppo_config.seq_length))
            print(f"values:\n{values.asnumpy()}")
            end_time = time.time()
            print("critic model end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

            self.ref_model.model.add_flags_recursive(use_past=False)
            start_time = time.time()
            print("reference model start at {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            ref_logprobs = self.ref_model(all_tokens, samples=all_tokens)
            print(f"ref_logprobs:\n{ref_logprobs.asnumpy()}")
            end_time = time.time()
            print("reference model end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

            logprobs = logprobs.asnumpy()
            values = values.asnumpy()
            ref_logprobs = ref_logprobs.asnumpy()

            n_samples: int = samples.shape[0]

            start = self.ppo_config.max_prompt_length - 1
            end = self.ppo_config.seq_length - 1
            valid_length_response = (samples.asnumpy() != self.ppo_config.pad_token_id).astype(int).sum(1) \
                - (prompt_tensors.asnumpy() != self.ppo_config.pad_token_id).astype(int).sum(1)
            print("=====all_logprobs", logprobs.shape, start, end)
            all_values = values[:, start:end]
            all_logprobs = logprobs[:, start:end]
            print("=====all_logprobs", all_logprobs, start, end)
            # kl_divergence_estimate = self.ppo_model.kl_ctl.value.asnumpy() * (logprobs - ref_logprobs) # Adaptive KL divergence
            kl_divergence_estimate = self.ppo_config.kl_coef * (logprobs - ref_logprobs) # Fixed KL divergence
            kl_divergence_estimate = kl_divergence_estimate[:, start:end]

            rollout_count = 0
            prompt_tensors = prompt_tensors.asnumpy()
            all_tokens = all_tokens.asnumpy()
            pretrain_ids = pretrain_ids.asnumpy()
            loss_mask = loss_mask.asnumpy() if type(loss_mask) is not np.ndarray else loss_mask

            start_time = time.time()
            print("rl element processing start at {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
            for sample_idx in range(n_samples):
                sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]
                rewards = sample_kl_divergence_estimate 
                
                all_logprobs[sample_idx][int(valid_length_response[sample_idx]): ] = 0.0
                all_values[sample_idx][int(valid_length_response[sample_idx]): ] = 0.0
                all_values = np.array(all_values).reshape((n_samples, -1))
                rewards[int(valid_length_response[sample_idx]): ] = 0.0
                print("all_logprobs0", all_logprobs)
                index = valid_length_response[sample_idx] if valid_length_response[sample_idx] < len(rewards) else -1
                if isinstance(scores, mindspore.Tensor):
                    scores = scores.asnumpy()
                rewards[int(index)-1] += scores[sample_idx] 

                response_length = len(rewards)

                lastgaelam = 0
                advantages_reversed = []
                for k in range(response_length):
                    t = response_length-k-1
                    nextvalues = all_values[sample_idx, t + 1] if t < response_length - 1 else 0.0
                    delta = rewards[t] + self.ppo_model.gamma * nextvalues - all_values[sample_idx, t]
                    lastgaelam = delta + self.ppo_model.gamma * self.ppo_model.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = np.stack(advantages_reversed[::-1])
                
                returns = advantages + all_values[sample_idx]

                # generate attention_mask
                response = all_tokens[sample_idx]
                attention_mask = np.not_equal(response, self.ppo_config.pad_token_id)
                max_prompt_length = self.ppo_config.max_prompt_length
                last_index = np.max(np.where(attention_mask > 0))
                # shift left 1 bit
                attention_mask[last_index] = 0.0
                attention_mask[:max_prompt_length-1] = 0.0
                print("all_logprobs[sample_idx]", all_logprobs, sample_idx, all_logprobs[sample_idx])
                ppo_rl_elements.append(
                    PPOData(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=all_tokens[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                        advantages=advantages,
                        returns=returns,
                        pretrain_ids=pretrain_ids[sample_idx],
                        loss_mask=loss_mask[sample_idx],
                        attention_mask=attention_mask,
                    )
                )

                rollout_count += 1
            end_time = time.time()
            print("rl element processing end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

        self.push_to_store(ppo_rl_elements)
        print("Avg scores:\n", np.mean(np.array(scores_record)), flush=True)

if __name__ == "__main__":
    # samples = np.random.randint(low=0, high=15, size=(10, 550)).astype(np.int32)
    # get_scores(samples)
    # reward_fn(samples)
    context.set_context(device_target='Ascend', device_id=1, mode=mindspore.GRAPH_MODE)
    trainer = PPOTrainer(ppo_config=PPOConfig)
    trainer.make_experience(num_rollouts=2)
