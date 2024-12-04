import time
import os
from dataclasses import dataclass, asdict
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context, ops
from mindspore.ops import operations as P
from mindspore.dataset import MindDataset
from mindspore.mindrecord import FileWriter
from mindspore.communication.management import get_rank

from mindformers import logger
from mindformers.trainer.utils import load_distributed_checkpoint

from mindrlhf.configs.ppo_configs import PPOConfig
from mindrlhf.models.reward_model import RewardModel, CriticModel
from mindrlhf.models.ppo_models import CausalLMHydraWithValueHead, PPO_model
from ..utils.utils import get_valid_length_each_example


@dataclass
class PPOData:
    query_tensors: Tensor
    response_tensors: Tensor
    logprobs: Tensor
    values: Tensor
    rewards: Tensor
    advantages: Tensor
    returns: Tensor
    pretrain_ids: Tensor
    loss_mask: Tensor
    attention_mask: Tensor


def get_first_diverge_indices(preferred_comp_ids, disfavored_comp_ids):
    is_equal = Tensor(preferred_comp_ids == disfavored_comp_ids).astype('float32')
    first_diverge_indices = is_equal.sum(axis=1, dtype=mindspore.int32)
    return first_diverge_indices


class RewardFn(nn.Cell):
    def __init__(self, model_config, load_ckpt_func=None):
        super(RewardFn, self).__init__()
        self.pad_token = model_config.pad_token_id
        self.reward_model = RewardModel(model_config)
        self.not_equal = P.NotEqual()

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
        self.sft_ckpt_path = sft_model_config.checkpoint_name_or_path
        sft_model_config.checkpoint_name_or_path = None
        self.ref_ckpt_path = ref_model_config.checkpoint_name_or_path
        ref_model_config.checkpoint_name_or_path = None
        self.critic_ckpt_path = critic_model_config.checkpoint_name_or_path
        critic_model_config.checkpoint_name_or_path = None
        self.reward_ckpt_path = rm_model_config.checkpoint_name_or_path
        rm_model_config.checkpoint_name_or_path = ""
        self.is_shared_backbone = ppo_config.is_shared_backbone
        self.mind_dataset_dir = ppo_config.mind_dataset_dir
        if self.mind_dataset_dir is not None:
            columns_to_project = ["prompt_ids", "pretrain_ids", "loss_mask"]
            mindspore.dataset.config.set_seed(2023)
            dataset = MindDataset(self.mind_dataset_dir).project(columns=columns_to_project)
            self.prompt_dataloader = dataset.take(ppo_config.num_rollouts)
            self.prompt_dataloader = self.prompt_dataloader.batch(batch_size=ppo_config.chunk_size
                                                                  * sft_model_config.parallel_config.data_parallel)
            self.prompt_iterator = self.prompt_dataloader.create_tuple_iterator()
        else:
            logger.info("In training stages, there is not dataset for making experience")

        self.sft_model_config = sft_model_config
        self.rm_model_config = rm_model_config
        policy_model = CausalLMHydraWithValueHead(sft_model_config, self.ppo_config)
        critic_model = None
        if not self.is_shared_backbone:
            critic_model = CriticModel(critic_model_config)
        self.ppo_model = PPO_model(ppo_config, policy_model, critic_model)
        self.ref_model = CausalLMHydraWithValueHead(ref_model_config, self.ppo_config)
        self.reward_fn = RewardFn(rm_model_config)
        self.ref_model.model.set_train(False)
        self.reward_fn.set_train(False)
        self.reward_fn.reward_model.set_train(False)
        self.reward_fn.reward_model.model.set_train(False)

        self.ref_mean = 0
        self.ref_std = 0
        self.cliprange_reward = 10.0
        self.store = []

        self.log_softmax = P.LogSoftmax(axis=-1)
        self.gather = P.GatherD()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=-1)
        self.depend = P.Depend()

    def load_checkpoint(self):
        load_ckpt_func = load_distributed_checkpoint if self.ppo_config.use_parallel else mindspore.load_checkpoint
        if self.sft_ckpt_path:
            param_dict = load_ckpt_func(self.sft_ckpt_path)
            # ============= different ckpt may not need to replace name =================
            # new_param_dict = {k.replace("transformer", "backbone").replace(
            #     "backbone.backbone", "backbone.transformer"): v for k, v in param_dict.items()}
            # ===========================================================================
            print(f"begin to load policy model from: {self.sft_ckpt_path}", flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.ppo_model.policy_model, param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)

        if self.critic_ckpt_path:
            param_dict = load_ckpt_func(self.critic_ckpt_path)
            # ============= different ckpt may not need to replace name =================
            # new_param_dict = {k.replace("reward_model.model.", "").replace("transformer", "backbone").replace(
            #     "backbone.backbone", "backbone.transformer"): v for k, v in param_dict.items()}
            # ===========================================================================
            print(f"begin to load critic model from: {self.critic_ckpt_path}", flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.ppo_model.critic_model, param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)

        if self.ref_ckpt_path:
            param_dict = load_ckpt_func(self.ref_ckpt_path)
            # ============= different ckpt may not need to replace name =================
            # new_param_dict = {k.replace("transformer", "").replace("transformer", "backbone").replace(
            #     "backbone.backbone", "backbone.transformer"): v for k, v in param_dict.items()}
            # ===========================================================================
            print(f"begin to load critic model from: {self.ref_ckpt_path}", flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.ref_model, param_dict)
            print(f"param not load: {param_not_load}", flush=True)
            print(f"ckpt not load: {ckpt_not_load}", flush=True)

        if self.reward_ckpt_path:
            param_dict = load_ckpt_func(self.reward_ckpt_path)
            print("Begin to load reward model ckpt from: ", self.reward_ckpt_path, flush=True)
            param_not_load, ckpt_not_load = mindspore.load_param_into_net(self.reward_fn.reward_model, param_dict)
            print("Parameter not loaded: ", param_not_load, flush=True)
            print("Ckpt not loaded: ", ckpt_not_load, flush=True)


    def push_to_store(self, data):
        self.store = data

    def save_checkpoint(self, rank_id=0, steps=0):
        save_dir = self.ppo_config.save_ckpt_dir + "/rank_{}".format(rank_id)
        if save_dir:
            print("Save checkpoints in {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ppo_filename = os.path.join(save_dir, "policy_model_epoch_{}.ckpt".format(steps))
            critic_filename = os.path.join(save_dir, "critic_model_epoch_{}.ckpt".format(steps))
            mindspore.save_checkpoint(self.ppo_model.policy_model, ppo_filename, integrated_save=False)
            if not self.is_shared_backbone:
                mindspore.save_checkpoint(self.ppo_model.critic_model, critic_filename, integrated_save=False)
        else:
            print("There is no checkpoint to save!")

    def save_ppoelement(self, save_path):
        if save_path:
            schema = {
                "query_tensors": {"type": "int32", "shape": [-1]},
                "response_tensors": {"type": "int32", "shape": [-1]},
                "logprobs": {"type": "float32", "shape": [-1]},
                "values": {"type": "float32", "shape": [-1]},
                "rewards": {"type": "float32", "shape": [-1]},
                "advantages": {"type": "float32", "shape": [-1]},
                "returns": {"type": "float32", "shape": [-1]},
                "pretrain_ids": {"type": "int32", "shape": [-1]},
                "loss_mask": {"type": "int32", "shape": [-1]},
                "attention_mask": {"type": "int32", "shape": [-1]},
            }

            writer = FileWriter(file_name=save_path, shard_num=1, overwrite=True)
            writer.add_schema(schema)
            count = 0
            for ele in self.store:
                count += 1
                print(asdict(ele))
                writer.write_raw_data([asdict(ele)])
            print("Total number of samples: {}".format(count))
            writer.commit()
            print("ppo element saved! Output file refer: {}".format(save_path))

    def generate(self, input_ids, attn_masks=None):
        print("input_ids", input_ids.shape)
        input_ids_numpy = input_ids.asnumpy()
        input_ids_list = input_ids_numpy.tolist()
        _, max_valid_length = get_valid_length_each_example(input_ids_numpy, self.ppo_model.pad_token_id)

        prompt_len = (np.array(input_ids_list) != self.ppo_config.pad_token_id).astype(int).sum(1)
        left_padding_prompt = np.ones((len(input_ids_list), self.ppo_config.max_prompt_length)
                                      ) * self.ppo_config.pad_token_id
        resposne_array = np.ones((len(input_ids_list), self.ppo_config.max_decode_length)) * \
            self.ppo_config.pad_token_id
        samples = np.ones((len(input_ids_list), self.ppo_config.seq_length)) * self.ppo_config.pad_token_id

        generate_begin_time = time.time()
        print("input_ids shape", input_ids.shape)
        outputs = self.ppo_model.policy_model.model.generate(input_ids_numpy[:, :max_valid_length], max_new_tokens=self.ppo_config.max_decode_length)
        print("Generating elapsed time: ", time.time() - generate_begin_time)
        for i in range(len(input_ids_list)):
            x = outputs[i][prompt_len[i]: prompt_len[i] + self.ppo_config.max_decode_length]
            resposne_array[i, :len(x)] = x
            print(resposne_array)
            p = outputs[i]
            samples[i, :len(p)] = p
            left_padding_prompt[i, self.ppo_config.max_prompt_length -
                                prompt_len[i]:] = input_ids_list[i][:prompt_len[i]]
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

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0, rank_id: int = 0):
        ep_begin_time = time.time()
        print("Make experience begin at {} \n------------------------------- "
              .format(time.strftime('%H:%M:%S', time.localtime(ep_begin_time))), flush=True)
        self.ppo_model.policy_model.model.set_train(False)
        if not self.is_shared_backbone:
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
            prompt_tensors = Tensor(batch[0], mstype.int32)
            pretrain_ids = Tensor(batch[1], mstype.int32)
            loss_mask = batch[2][:, 1:]

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
            self.ppo_model.policy_model.model.add_flags_recursive(is_first_iteration=True)

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

            if self.is_shared_backbone:
                logprobs, values = self.ppo_model.policy_model(all_tokens, samples=all_tokens, return_value=True)
            else:
                logprobs = self.ppo_model.policy_model(all_tokens, samples=all_tokens)
            end_time = time.time()
            print("policy model end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

            if not self.is_shared_backbone:
                start_time = time.time()
                print("critic model start at {}-------------------------------"
                      .format(time.strftime('%H:%M:%S', time.localtime(start_time))), flush=True)
                values = self.ppo_model.critic_model(all_tokens)
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
            values = values.asnumpy().astype(np.float32)
            ref_logprobs = ref_logprobs.asnumpy()
            n_samples = samples.shape[0]

            start = self.ppo_config.max_prompt_length - 1
            end = self.ppo_config.seq_length - 1
            valid_length_response = (samples.asnumpy() != self.ppo_config.pad_token_id).astype(int).sum(1) \
                - (prompt_tensors.asnumpy() != self.ppo_config.pad_token_id).astype(int).sum(1)
            all_values = values[:, start:end]
            all_logprobs = logprobs[:, start:end]
            kl_divergence_estimate = self.ppo_config.kl_coef * (logprobs - ref_logprobs)  # Fixed KL divergence
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
                all_logprobs[sample_idx][int(valid_length_response[sample_idx]):] = 0.0
                all_values[sample_idx][int(valid_length_response[sample_idx]):] = 0.0
                all_values = np.array(all_values).reshape((n_samples, -1))
                rewards[int(valid_length_response[sample_idx]):] = 0.0
                index = valid_length_response[sample_idx] if valid_length_response[sample_idx] < len(rewards) else 0
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
                        query_tensors=prompt_tensors[sample_idx].astype(np.int32),
                        response_tensors=all_tokens[sample_idx].astype(np.int32),
                        logprobs=all_logprobs[sample_idx].astype(np.float32),
                        values=all_values[sample_idx].astype(np.float32),
                        rewards=rewards.astype(np.float32),
                        advantages=advantages.astype(np.float32),
                        returns=returns.astype(np.float32),
                        pretrain_ids=pretrain_ids[sample_idx].astype(np.int32),
                        loss_mask=loss_mask[sample_idx].astype(np.int32),
                        attention_mask=attention_mask.astype(np.int32),
                    )
                )

                rollout_count += 1
            end_time = time.time()
            print("rl element processing end at {}, elapsed time {}-------------------------------"
                  .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-start_time), flush=True)

        self.push_to_store(ppo_rl_elements)
        print("Avg scores:\n", np.mean(np.array(scores_record)), flush=True)
        end_time = time.time()
        print("Make experience, end at {}, elapsed time {} \n------------------------------- "
              .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-ep_begin_time), flush=True)
        if self.ppo_config.save_data_file:
            if rank_id % 8 == 0:
                self.save_ppoelement(self.ppo_config.save_data_file)
        self.ppo_model.policy_model.set_train()
        if not self.is_shared_backbone:
            self.ppo_model.critic_model.set_train()

    def train(self, ppo_with_grad, dataset):
        sink_process = mindspore.data_sink(ppo_with_grad, dataset, sink_size=self.ppo_config.sink_size)
        steps = dataset.dataset_size // self.ppo_config.sink_size
        print(f"dataset size is {dataset.dataset_size}, sink size is {self.ppo_config.sink_size},"
              f"total steps is {steps}")
        for epoch in range(self.ppo_config.epochs):
            ep_begin_time = time.time()
            print("Epoch {}, begin at {} \n------------------------------- "
                  .format(epoch+1, time.strftime('%H:%M:%S', time.localtime(ep_begin_time))), flush=True)
            for batch in range(steps):
                for i in range(self.ppo_config.ppo_epochs):
                    out = sink_process()
                    print("PPO Batch: {} | PPO Epoch: {} | loss: {} | lr: {} | is overflow: {} | loss scale: {}"
                          .format(batch, i, out[0], out[1], out[2], out[3]), flush=True)
            end_time = time.time()
            print("Epoch {}, end at {}, elapsed time {} \n------------------------------- "
                  .format(epoch+1, time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-ep_begin_time), flush=True)
        # save checkpoint after each training
        self.save_checkpoint(rank_id=get_rank(), steps=epoch)

    def pre_run(self, stage_name):
        if self.ppo_config.only_save_strategy:
            if context.get_auto_parallel_context("parallel_mode") in ['semi_auto_parallel', 'auto_parallel',
                                                                      'hybrid_parallel']:
                batch_size = self.ppo_config.batch_size * self.ppo_config.parallel_config.get("data_parallel", 1)
                fake_data = ops.zeros((batch_size, self.ppo_config.seq_length), mstype.int32)
                context.set_auto_parallel_context(strategy_ckpt_config=
                                                  {"save_file":
                                                       f"./{stage_name}_policy_strategy/strategy_{get_rank()}.ckpt"})
                self.ppo_model.policy_model.add_flags_recursive(use_past=False)
                self.ppo_model.policy_model.add_flags_recursive(is_first_iteration=True)
                self.ppo_model.policy_model.compile(fake_data, samples=fake_data, return_value=True)
                context.set_auto_parallel_context(strategy_ckpt_config=
                                                  {"save_file":
                                                       f"./{stage_name}_reward_strategy/strategy_{get_rank()}.ckpt"})
                self.reward_fn.compile(fake_data)
                context.set_auto_parallel_context(strategy_ckpt_config=
                                                  {"save_file":
                                                       f"./{stage_name}_ref_strategy/strategy_{get_rank()}.ckpt"})
                self.ref_model.compile(fake_data, samples=fake_data)
                logger.info("Running only save strategy finish, system exit.")
                exit(0)
            else:
                logger.info("only_save_strategy is True, but stand_alone and data_parallel mode"
                            "do not have strategy file, system exit!")
                exit(0)


if __name__ == "__main__":
    context.set_context(device_target='Ascend', device_id=1, mode=mindspore.GRAPH_MODE)
    trainer = PPOTrainer(ppo_config=PPOConfig)
    trainer.make_experience(num_rollouts=2)
