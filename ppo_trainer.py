from dataclasses import dataclass
import numpy as np
import mindspore
from mindspore import Tensor, ops, nn, ms_function
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype

@dataclass
class PPOElement:
    query_tensor: Tensor
    sample_tensor: Tensor
    logprobs: Tensor
    values: Tensor
    rewards: Tensor

class RewardScores(nn.Cell):
    def __init__(self, rm_config, reward_model):
        super(RewardScores, self).__init__()
        rm_config.dropout_rate = float(0)
        self.config = rm_config
        self.reward_model = reward_model
        self.reward_model.set_train(False)
        self.PAD_ID = rm_config.eos_token

    def get_end_indices(self, comp_input_ids, batch_size):
        not_padding = F.cast(comp_input_ids != self.PAD_ID, mindspore.float32)
        first_pad_indices = not_padding.sum(axis=1)
        # print("first_pad_indices", first_pad_indices)
        preferred_end_indices = (first_pad_indices[:batch_size] - 1).reshape(1, batch_size)
        disfavored_end_indices = (first_pad_indices[batch_size:] - 1).reshape(1, batch_size)
        comp_end_indices = F.concat((preferred_end_indices, disfavored_end_indices), axis=0)
        max_end_indices = F.cast(comp_end_indices.max(axis=0), mindspore.int32)
        return max_end_indices

    def get_scores(self, samples):
        batch_size = 2
        scores_list = []
        samples = F.cast(samples, mindspore.int32)
        for i in range(0, len(samples), batch_size):
            input_ids = samples[i : i + batch_size]
            attn_masks = (input_ids != self.PAD_ID).to(mindspore.float32)
            comp_input_ids = F.concat((input_ids, input_ids), axis=0)
            comp_attn_masks = F.concat((attn_masks, attn_masks), axis=0)
            end_indices = self.get_end_indices(comp_input_ids, batch_size)
            bs_scores = self.reward_model.infer(comp_input_ids, comp_attn_masks, end_indices)
            for j in range(batch_size):
                scores_list.append(bs_scores[j])
        scores = F.stack(scores_list, axis=0)
        return scores

    def construct(self, samples, original_samples):
        original_scores = self.get_scores(original_samples)
        scores = self.get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores


# class MindPPOTrainer(nn.Cell):
class MindPPOTrainer():
    def __init__(self,
                config,
                rm_config,
                prompt_dataset,
                val_dataset,
                ppo_model,
                ref_model,
                reward_model,
                optimizer,
                ppo_with_grad):
        super(MindPPOTrainer, self).__init__()
        self.prompt_iterator = prompt_dataset
        self.val_iterator = val_dataset
        self.config = config
        self.ppo_model = ppo_model
        # convert ckpt for sucessfully loading
        self.model = self.ppo_model.policy_model # backbone
        self.ref_model = ref_model
        self.ref_model.set_train(False)
        self.optimizer = optimizer
        self.ppo_with_grad = ppo_with_grad
        self.ref_mean = 0
        self.ref_std = 0
        self.cliprange_reward = 10.0
        self.ppo_elements = []
        self.seq_length = self.config.seq_length
        self.max_prompt_length = self.config.max_prompt_length
        self.max_decode_length = self.config.max_decode_length
        self.pad_token_id = self.config.pad_token_id
        self.chunk_size = self.config.chunk_size
        self.reward_scores = RewardScores(rm_config, reward_model)
        self.offset = Tensor(np.arange(self.max_decode_length) - 1)
        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=1)
        self.gather = ops.Gather(-1)
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gatherd = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)

    def logprobs_of_labels(self, logits, labels):
        """Compute log probs of the labels"""
        logprobs = self.log_softmax(logits)
        logprobs_labels = self.gatherd(logprobs, -1, self.unsqueeze(labels, -1))
        return self.squeeze(logprobs_labels)

    def gather_ppo_elements(self, data):
        self.ppo_elements = data

    def generate(self, input_ids, attn_masks=None):
        """generate response"""
        input_ids_list = input_ids.asnumpy().tolist()
        prompt_len = (np.array(input_ids_list) != self.pad_token_id).astype(int).sum(1)
        left_padding_prompt = np.ones((len(input_ids_list), self.config.max_prompt_length)) * self.config.pad_token_id
        right_padding_prompt = np.ones((len(input_ids_list), self.config.max_prompt_length)) * self.config.pad_token_id # right padding for test
        resposne_array = np.ones((len(input_ids_list), self.max_decode_length)) * self.pad_token_id
        samples = np.ones((len(input_ids_list), self.seq_length)) * self.pad_token_id
        right_padding_prompt = input_ids.asnumpy()
        # outputs = self.model.do_generate(right_padding_prompt)
        outputs = self.model.do_generate_graph(input_ids)
        outputs = outputs.asnumpy()
        for i in range(len(input_ids_list)):
            x = outputs[i][prompt_len[i]: prompt_len[i] + self.config.max_decode_length]
            resposne_array[i, :len(x)] = x
            p = outputs[i]
            samples[i, :len(p)] = p
            left_padding_prompt[i, self.config.max_prompt_length - prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]
        return Tensor(samples, mstype.int32), Tensor(resposne_array, mstype.int32), Tensor(left_padding_prompt, mstype.int32)

    @ms_function
    def generate_element(self, prompt_tensors, samples, responses, original_samples):
        """generate ppo element"""
        all_scores = self.reward_scores(samples, original_samples=original_samples)
        scores = all_scores
        prompt_tensors = self.cast(prompt_tensors, mstype.int32)
        sample_outputs = self.cast(responses, mstype.int32)
        all_tokens = self.concat((prompt_tensors, sample_outputs))
        attention_mask = (all_tokens != self.pad_token_id)
        attention_mask = self.cast(attention_mask, mstype.float32)
        logits, values, _ = self.model(all_tokens, attention_mask)
        ref_logits, _, _ = self.ref_model(all_tokens, attention_mask)
        logprobs = self.logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
        ref_logprobs = self.logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])
        values = values[:, :-1]
        starts = self.max_prompt_length - 1
        ends = self.seq_length - 1
        valid_length_response = (responses != self.pad_token_id).to(mstype.int32).sum(axis=-1)
        all_values = values[:, starts:ends]
        all_logprobs = logprobs[:, starts:ends]
        kl_divergence_estimate = self.ppo_model.kl_ctl.value * (logprobs - ref_logprobs)
        kl_divergence_estimate = kl_divergence_estimate[:, starts:ends]
        res_atten_mask = attention_mask[:, starts:]
        res_atten_mask = res_atten_mask[:, :self.max_decode_length]
        all_logprobs = ops.mul(all_logprobs, res_atten_mask)
        all_values = ops.mul(all_values, res_atten_mask)
        return valid_length_response, scores, kl_divergence_estimate, all_logprobs, all_values, all_tokens

    def generate_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        self.ppo_model.set_train(False)
        self.ref_model.set_train(False)
        ppo_rl_elements = []
        while len(ppo_rl_elements) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
                batch = next(self.prompt_iterator)
            prompt_tensors = self.cast(batch[0], mstype.int32)
            original_samples =self.cast(batch[2], mstype.int32)
            # padded_samples, responses = self.generate_graph(prompt_tensors) # responses: right padding
            padded_samples, responses, prompt_tensors = self.generate(prompt_tensors) # responses: right padding

            n_samples = padded_samples.shape[0]
            valid_length_response, scores, kl_divergence_estimate, all_logprobs, all_values, all_tokens = \
                self.generate_element(prompt_tensors, padded_samples, responses, original_samples)

            padded_samples = padded_samples.asnumpy()
            responses = responses.asnumpy()

            rollout_count = 0
            valid_length_response = valid_length_response.asnumpy()
            scores = scores.asnumpy()
            kl_divergence_estimate = kl_divergence_estimate.asnumpy()
            all_values = all_values.asnumpy()
            all_tokens = all_tokens.asnumpy()
            prompt_tensors = prompt_tensors.asnumpy()
            for sample_idx in range(n_samples):
                sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]
                rewards = sample_kl_divergence_estimate
                index = valid_length_response[sample_idx] if valid_length_response[sample_idx] < len(rewards) else -1
                rewards[index] += scores[sample_idx]
                logprobs_f = all_logprobs[sample_idx]
                values_f = all_values[sample_idx]
                rewards_f = rewards
                ppo_rl_elements.append(
                    PPOElement(
                        query_tensor=prompt_tensors[sample_idx],
                        sample_tensor=all_tokens[sample_idx],
                        logprobs=logprobs_f,
                        values=values_f,
                        rewards=rewards_f,
                    )
                )
                rollout_count += 1
        print("ppo_rl_elements size", len(ppo_rl_elements))
        self.gather_ppo_elements(ppo_rl_elements)

    @ms_function
    def generate_graph(self, prompt_tensors, attention_mask=None):
        if not attention_mask:
            attention_mask = self.cast(prompt_tensors != self.pad_token_id, mstype.float32)
        prompt_len = self.sum(attention_mask, 1)
        prompt_len_end = prompt_len + self.max_decode_length
        samples = self.model.do_generate_graph(prompt_tensors)
        resposne_array = []
        for i in range(samples.shape[0]):
            input_indices = self.cast(ops.arange(prompt_len[i], prompt_len_end[i]), mstype.int32)
            x = self.gather(samples[i], input_indices, 0)
            x = self.unsqueeze(x, 0)
            resposne_array.append(x)
        resposne_array = F.concat(resposne_array)
        return samples, resposne_array

    def evaluate(self):
        self.ppo_model.set_train(False)
        self.reward_scores.set_train(False)
        reward_mean = 0
        n = 0
        for prompts, _, original_samples , _ in self.val_iterator:
            prompts = prompts.to(mstype.int32)
            original_samples = original_samples.to(mstype.int32)
            samples, _ = self.generate_graph(prompts)
            reward_mean += self.eval_scores(samples, original_samples)
            n += 1
            if n > 100:
                break
        print("self.val_dataloader dataset size", n)
        reward_mean = reward_mean / n
        self.ppo_model.set_train(True)
        return reward_mean

    @ms_function
    def eval_scores(self, samples, original_samples):
        all_scores = self.reward_scores(samples, original_samples)
        reward_mean = self.mean(all_scores)
        return reward_mean

    def train(self, dataset):
        self.ppo_with_grad.set_train()
        n = dataset.get_dataset_size()
        iterator = dataset.create_dict_iterator()
        loss = Tensor(0)
        for batch_id, databatch in enumerate(iterator):
            for _ in range(self.config.ppo_epochs):
                query_tensors = databatch["query_tensors"].to(mstype.int32)
                sample_tensors = databatch["sample_tensors"].to(mstype.int32)
                old_logprobs = databatch["logprobs"].to(mstype.float32)
                old_values = databatch["values"].to(mstype.float32)
                old_rewards = databatch["rewards"].to(mstype.float32)
                attention_mask = sample_tensors != self.config.pad_token_id
                attention_mask = ops.Cast()(attention_mask, mstype.float32)
                loss, approx_kl = self.ppo_with_grad(query_tensors, sample_tensors, old_logprobs, old_values, old_rewards, attention_mask)
            if batch_id % 2 == 0:
                print(f"loss: {loss.asnumpy():>7f}  [{batch_id:>3d}/{n:>3d}]")
            _ = self.ppo_model.post_backward_callback(approx_kl, self.config.batch_size)

# def train_loop(config, dataset, ppo_model, ppo_with_grad):
#     ppo_with_grad.set_train()
#     n = dataset.get_dataset_size()
#     iterator = dataset.create_dict_iterator()
#     loss = Tensor(0)
#     for batch_id, databatch in enumerate(iterator):
#         for _ in range(config.ppo_epochs):
#             query_tensors = databatch["query_tensors"].to(mstype.int32)
#             sample_tensors = databatch["sample_tensors"].to(mstype.int32)
#             old_logprobs = databatch["logprobs"].to(mstype.float32)
#             old_values = databatch["values"].to(mstype.float32)
#             old_rewards = databatch["rewards"].to(mstype.float32)
#             attention_mask = sample_tensors != config.pad_token_id
#             attention_mask = ops.Cast()(attention_mask, mstype.float32)
#             loss, approx_kl = ppo_with_grad(query_tensors, sample_tensors, old_logprobs, old_values, old_rewards, attention_mask)
#         if batch_id % 2 == 0:
#             print(f"loss: {loss.asnumpy():>7f}  [{batch_id:>3d}/{n:>3d}]")
#         _ = ppo_model.post_backward_callback(approx_kl, config.batch_size)
