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

"""
For text generation
"""
from typing import Optional, List, Union
import numpy as np

from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import context
from mindformers.generation.streamers import BaseStreamer

from mindrlhf.utils.utils import set_pipeline_parallel_context

__all__ = ['GeneratorMixin']

def topk_fun(logits, topk=5):
    """Get topk"""
    batch_value = []
    batch_index = []
    for i in range(logits.shape[0]):
        target_column = logits[i].tolist()
        sorted_array = [(k, v) for k, v in enumerate(target_column)]
        sorted_array.sort(key=lambda x: x[1], reverse=True)
        topk_array = sorted_array[:topk]
        index, value = zip(*topk_array)
        batch_value.append(value)
        batch_index.append(index)
    return np.array(batch_value), np.array(batch_index)


def batch_select(data, index):
    """bathc operation to sorted_logits[:, :top_p_num]"""
    output = []
    for i in range(data.shape[0]):
        res = data[i, :index[i]]
        output.append(res.reshape(1, -1))
    return np.concatenate(output, 0)


def sampler(log_probs_revised, top_p, top_k, use_pynative=False):
    """Convert the log_probs to probability"""
    if use_pynative:
        logits = P.Pow()(np.e, Tensor(log_probs_revised, mstype.float32))
    else:
        logits = np.power(np.e, np.array(log_probs_revised, np.float32))

    # If top_p is less than 1.0, use top_p sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        if use_pynative:
            sorted_logits, index = P.TopK(sorted=True)(logits, 5000)
            cumsum_logits = P.CumSum()(sorted_logits, 1)
            cumsum_logits = cumsum_logits.asnumpy()
            index = index.asnumpy()
            sorted_logits = sorted_logits.asnumpy()
        else:
            sorted_logits, index = topk_fun(logits, 5000)
            cumsum_logits = np.cumsum(sorted_logits, 1)
        cumsum_logits = cumsum_logits
        index = index
        sorted_logits = sorted_logits
        top_p_num = np.sum(cumsum_logits < top_p, axis=-1) + 1
        # Get the corresponding probs and indices
        probs = batch_select(sorted_logits, top_p_num)
        p_args = batch_select(index, top_p_num)
        p = probs / np.sum(probs, -1, keepdims=True)
        # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        if use_pynative:
            probs, p_args = P.TopK(sorted=True)(logits, top_k)
            probs = probs.asnumpy()
            p_args = p_args.asnumpy()
        else:
            probs, p_args = topk_fun(logits, top_k)
        probs = probs
        p_args = p_args
        # Avoid rounding error
        for i in range(probs.shape[0]):
            if np.sum(probs[i]) == 0:
                probs[i] = np.array([1 / top_k for _ in range(top_k)])
        p = probs / np.sum(probs, -1, keepdims=True)
    return p, p_args


def precision_correct(p, top_p, top_k, batch_size):
    # Avoid rounding error
    if top_p == 1:
        for i in range(batch_size):
            if np.sum(p[i]) == 0:
                p[i] = np.array([1 / top_k for _ in range(top_k)])
        p = p / np.sum(p, -1, keepdims=True)
    return p


class GeneratorMixin:
    """Generator For the nlp models"""
    def __init__(self):
        pass

    def _pad_inputs_using_max_length(self, origin_inputs, pad_token_id=0):
        input_ids = pad_token_id * np.ones((len(origin_inputs), self.ppo_config.seq_length), dtype="int32")
        for i, origin_input in enumerate(origin_inputs):
            ilen = len(origin_input)
            if ilen > self.ppo_config.seq_length:
                raise ValueError(f"origin_inputs size is {ilen}, you should increase the "
                                 f"seq_length of the model {self.ppo_config.seq_length}.")
            input_ids[i, :ilen] = origin_input
        return input_ids

    def generate_pos_id_and_mask_for_incr_infer(self, **kwargs):
        """should be implemented in pretrained model,
        if you need to generate position ids and attention mask before construct.
        here gives None by default"""
        _ = kwargs
        return None, None

    def _incremental_infer(self,
                           input_ids,
                           current_index,
                           valid_length_each_example,
                           is_first_iteration,
                           position_ids=None,
                           attention_mask=None):
        """model forward for incremental infer."""
        # Claim the first graph
        if self.policy_model.model.is_first_iteration:
            self.policy_model.model.add_flags_recursive(is_first_iteration=True)
            attention_mask_tmp = None
            if attention_mask:
                attention_mask_tmp = Tensor(attention_mask, mstype.float32)
            print("===== first iteration: ", input_ids.tolist(), current_index, attention_mask_tmp, valid_length_each_example, self.policy_model.model.is_first_iteration, self.policy_model.model.use_past, flush=True)
            log_probs = self.policy_model(
                input_ids=Tensor(input_ids, mstype.int32),
                input_position=Tensor(current_index, mstype.int32),
                attention_mask=attention_mask_tmp,
                init_reset=Tensor([False], mstype.bool_),
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32),
                is_first_iteration=self.policy_model.model.is_first_iteration, 
                use_past=self.policy_model.model.use_past,
            )
            print("===== first iteration output: ", log_probs, flush=True)
            # first iter done, go to other iters
            self.policy_model.model.is_first_iteration = False
        else:
            self.policy_model.model.add_flags_recursive(is_first_iteration=False)

            inputs_tmp = []
            for i in range(len(current_index)):
                current_index_tmp = int(current_index[i]) - i * input_ids.shape[1] # multibatch by huangziling
                # use numpy to slice array to avoid complie ascend slice op
                inputs_tmp.append(input_ids[i][current_index_tmp:current_index_tmp + 1])
            inputs_tmp = np.array(inputs_tmp, dtype=np.int32)

            attention_mask_tmp = None
            if attention_mask:
                attention_mask_tmp = attention_mask[:, :, current_index_tmp:current_index_tmp + 1, :]
                attention_mask_tmp = Tensor(attention_mask_tmp, mstype.float32)

            # print("===== other iterations: ", inputs_tmp, current_index, attention_mask_tmp, valid_length_each_example, self.policy_model.model.is_first_iteration, self.policy_model.model.use_past, flush=True)
            
            log_probs = self.policy_model(
                input_ids=Tensor(inputs_tmp, mstype.int32),
                input_position=Tensor(current_index, mstype.int32),
                attention_mask=attention_mask_tmp,
                init_reset=Tensor([True], mstype.bool_),
                batch_valid_length=Tensor([valid_length_each_example], mstype.int32),
                is_first_iteration=self.policy_model.model.is_first_iteration, 
                use_past=self.policy_model.model.use_past,
                # batch_valid_length (1,) int32 5
            )
            # print("===== other iterations output: ", log_probs, flush=True)

        return log_probs

    def _forward(self,
                 origin_inputs,
                 top_k,
                 top_p,
                 repetition_penalty,
                 max_length,
                 eos_token_id,
                 streamer=None,
                 pad_token_id=None):
        """
        Text generation given the model and origin inputs

        Inputs:
            model: The model to run the prediction
            end_token(int): The model will stop generating the words when it reaches the end_token.
            origin_inputs(list): The prompt for generation, should be a list of ids.
            model_origin_max_length(int): The sequence length of the model trained.
            max_length(int):  The maximum of generated length.
            vocab_size(int): The vocabulary length of the model.
            config: Inference configurations.
            streamer: Streamer object that will be used to stream the generated sequences.

        Returns:
            outputs: the ids for the generated text
        """
        if pad_token_id is None:
            pad_token_id = 0
        # Get configurations for inference
        use_pynative = False

        if streamer is not None:
            streamer.put(origin_inputs[0])

        batch_size = len(origin_inputs)

        print("The input shape is: %s", origin_inputs.shape)
        valid_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(origin_inputs[i] != pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)
        print("Get the valid for each example is: %s", valid_length_each_example)
        if np.max(valid_length_each_example) > max_length:
            raise ValueError("The max_length set is smaller than the length in the input_ids. You shout set "
                             f"max_length to {np.max(valid_length_each_example)}")

        target_length = [self.ppo_config.seq_length if valid_length_each_example[i] + self.ppo_config.max_decode_length \
            > self.ppo_config.seq_length else valid_length_each_example[i] + self.ppo_config.max_decode_length for i in range(batch_size)]
        print("max target_length is: %s", target_length)
        # A list of the frequency of each token
        frequency_list = None
        input_ids = self._pad_inputs_using_max_length(origin_inputs=origin_inputs, pad_token_id=pad_token_id)

        print("pad the origin inputs from %s into shape: %s", origin_inputs.shape, input_ids.shape)

        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1
        encoder_output = None
        encoder_mask = None

        # A single loop generates one token, loop until reaching target model_origin_max_length or generating eod token
        is_finished = [False] * batch_size

        # setup is_first_iteration flag for incremental infer
        if self.policy_model.model.use_past:
            self.policy_model.model.is_first_iteration = True
        is_first_iteration = False
        count = 0
        while np.sum(is_finished) != batch_size:
            if count == 4:
                break
            seq_length = input_ids.shape[1]

            # current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]

            current_index = [(valid_length_each_example[i+j*batch_size] - 1 + i * seq_length) \
                for i in range(batch_size) for j in range(self.ppo_config.inference_micro_size)]
            print("===== current_index: ", current_index, flush=True)

            current_index = Tensor(current_index, mstype.int32)
            # print("validate length: %s", valid_length_each_example)
            if self.policy_model.model.use_past:
                # print("===== use_past", flush=True)
                is_first_iteration = self.policy_model.model.is_first_iteration
                # generate input_position & attention_mask for incremental
                position_ids, attention_mask = self.generate_pos_id_and_mask_for_incr_infer(
                    input_ids=input_ids,
                    current_index=current_index,
                    valid_length_each_example=valid_length_each_example
                )
                # incremental generate
                log_probs = self._incremental_infer(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    current_index=current_index,
                    valid_length_each_example=valid_length_each_example,
                    is_first_iteration=is_first_iteration
                )
            else:
                print("===== Not use_past", flush=True)
                # auto-aggressive generate
                # log_probs = self.policy_model(Tensor(input_ids, mstype.int32), current_index)[0]

                # for multi-microbatch inference
                print("===== input_ids & current_index: ", input_ids, input_ids.shape, current_index, current_index.shape, flush=True)
                
                # log_probs = self.inference_wrapper(Tensor(input_ids, mstype.int32), current_index).squeeze()
                log_probs = self.policy_model(Tensor(input_ids, mstype.int32), current_index)

            # print("===== log_probs: ", np.shape(log_probs.asnumpy()), log_probs, flush=True)
            if self.policy_model.model_config.parallel_config.pipeline_stage > 1:
                context.reset_auto_parallel_context()
                log_probs = self.sr_net_logprobs(log_probs)
                set_pipeline_parallel_context(parallel_mode=self.ppo_config.parallel_mode, full_batch=self.ppo_config.full_batch,
                    optimizer_shard=self.policy_model.model_config.parallel_config.optimizer_shard,
                    stage_num=self.policy_model.model_config.parallel_config.pipeline_stage, 
                    enable_alltoall=self.ppo_config.enable_alltoall) 

            # Sample
            log_probs = log_probs.asnumpy()
            vocab_size = log_probs.shape[-1]
            if repetition_penalty != 1 and frequency_list is None:
                frequency_list = np.array([[0 for _ in range(vocab_size)]])
            log_probs_revised = log_probs.reshape(batch_size, vocab_size)
            if repetition_penalty != 1:
                log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                    (frequency_list > 0) * repetition_penalty
                  
            # p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)
            p = np.ones_like(log_probs)
            p_args = log_probs

            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                target_index = np.random.choice(len(p[i]), p=p[i])

                # update frequency list
                target = p_args[i][target_index]
                print(f"Select: {target}", flush=True)

                if repetition_penalty != 1:
                    frequency_list[0][target] = frequency_list[0][target] + 1
                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                if streamer is not None:
                    streamer.put(np.asarray([target]))
                    
                valid_length_each_example[i] += int(1)
                input_mask[i][valid_length_each_example[i] - 1] = 1

                # Stop judgment
                if p_args[i][target_index] == eos_token_id or valid_length_each_example[i] == target_length[i]:
                    is_finished[i] = True
                    continue
            count += 1

        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))

        if streamer is not None:
            streamer.end()
        return output_ids

    def generate(self,
                 input_ids: Optional[Union[List[int], List[List[int]]]],
                 do_sample: Optional[bool] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 max_length: Optional[int] = None,
                 streamer: Optional[BaseStreamer] = None):
        """
        Generate the words according to the given the input ids.

        Args:
            input_ids(List(str), List(List(str))): The token id list or a list of token id list.
            do_sample(bool): Whether do sampling on the candidate ids. If set True it will be enabled, and set it to be
                False to disable the sampling, equivalent to topk 1. If set None, it follow the setting in the
                configureation in the model. Default None.
            top_k(int): Determine the topK numbers token id as candidate. This should be a positive number.
                If set None, it follows the setting in the configureation in the model. Default None.
            top_p(float): The accumulation probability of the candidate token ids below the top_p will be select as the
                condaite ids. The validate the value of top_p is between (0, 1]. If the value is larger than 1,
                top_K algorithm will be enabled. If set None, it follow the setting in the configureation in the model.
                Default None.
            eos_token_id(int): The end of sentence token id. If set None, it follow the setting in the configureation
                in the model. Default None.
            repetition_penalty(float): The penalty factor of the frequency that generated words. The If set 1,
                the repetition_penalty will not be enabled. If set None, it follow the setting in the configureation in
                the model. Default None.
            max_length: The maximum length of the generated words. If set None, it follow the setting in the
                configureation in the model. Default None.
            streamer: The streamer that generator uses.


        Examples:
            >>> from mindformers import T5ForConditionalGeneration, T5Tokenizer
            >>> t5 = T5ForConditionalGeneration.from_pretrained("t5_small")
            >>> tokenizer = T5Tokenizer.from_pretrained("t5_small")
            >>> words = "translate the English to the Romanian: UN Chief Says There Is No Military Solution in Syria"
            >>> words = tokenizer(words, max_length=21, padding='max_length')['input_ids']
            >>> output = t5.generate(words, do_sample=True)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            eful ONU declară că nu există o soluţie militară în Siria
            >>> # Enable the top p sampling
            >>> output = t5.generate(words, do_sample=True, top_p=0.4)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            eful ONU declară că nu există o soluţie militară în Siria
            >>> # Enable the top k sampling.
            >>> output = t5.generate(words, do_sample=True, top_k=10, top_p=1)
            >>> output = tokenizer.decode(output[0], skip_special_tokens=True)
            >>> print(output)
            Este comist de stat ale stateului membre nai uzusepa şi ONU

        Returns:
            A list of the generated token ids
        """
        origin_phase = self.policy_model.model.phase
        self.policy_model.model.set_train(False)
        input_ids = np.array(input_ids).reshape(-1, np.shape(input_ids)[-1])
        config = self.ppo_config
        top_p = config.top_p if top_p is None else top_p
        top_k = config.top_k if top_k is None else top_k
        repetition_penalty = config.repetition_penalty if repetition_penalty is None else repetition_penalty
        max_length = config.max_decode_length if max_length is None else max_length
        eos_token_id = config.eos_token_id if eos_token_id is None else eos_token_id
        pad_token_id = config.pad_token_id if pad_token_id is None else pad_token_id
        do_sample = config.do_sample if do_sample is None else do_sample

        if not do_sample:
            top_p = 1
            top_k = 1
        # eval ops

        output_ids = self._forward(origin_inputs=input_ids,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   max_length=max_length,
                                   eos_token_id=eos_token_id,
                                   pad_token_id=pad_token_id,
                                   streamer=streamer)

        # print("The output is: ", tokenizer.decode(output_ids[0]), tokenizer.decode(output_ids[1]), flush=True)

        # set to original phase
        self.policy_model.model.set_train(origin_phase == 'train')
        return output_ids
