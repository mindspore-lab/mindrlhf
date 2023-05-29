from typing import Optional, List, Union
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import ops
from mindspore.common.tensor import Tensor


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

class ProcessLogits(nn.Cell):
    def __init__(self):
        super(ProcessLogits, self).__init__()
        self.e = Tensor(np.e)
        self.gather = P.Gather()
        self.logsoftmax = P.LogSoftmax()
        self.reshape = P.Reshape()

    def construct(self, logits, current_index=None):
        """Process the logits"""
        if current_index is not None:
            index = current_index.view(-1,)
            if len(logits.shape) == 3:
                logits = self.reshape(logits, (logits.shape[0]*logits.shape[1], -1))
            logits = self.gather(logits, index, 0)
        outputs = self.logsoftmax(logits)
        outputs = F.tensor_pow(self.e, outputs)
        return outputs

class GeneratorMixin:
    """Generator For the nlp models"""
    def __init__(self):
        self.e = Tensor(np.e)
        self.process_logits = ProcessLogits()

    def _prepare_model_inputs_for_decoder(self, input_ids, input_mask):
        """generate the inputs for the decoder"""
        batch_size = input_ids.shape[0]

        encoder_mask = Tensor(input_mask, mstype.float32)

        encoder_output = self.encoder_forward(Tensor(input_ids, mstype.int32),
                                              encoder_mask)

        input_ids = np.zeros((batch_size, self.config.max_decode_length))
        print("Decoder: pad the origin inputs into shape: %s", input_ids.shape)
        target_mask = np.zeros_like(input_ids)
        target_mask[:, 0] = 1

        # As the decoder is generating from [START] token
        return encoder_output, encoder_mask, input_ids, target_mask

    def _pad_inputs_using_max_length(self, origin_inputs):
        # pad the input_ids to the max_length
        pad_length = self.config.seq_length - origin_inputs.shape[-1]
        if pad_length < 0:
            raise ValueError(f"origin_inputs size is {origin_inputs.shape}, you should increase the "
                             f"seq_length of the model {self.config.seq_length}.")
        # Pad original inputs to model_origin_max_length
        input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))

        return input_ids

    # # @ms_function
    # def process_logits(self, logits, current_index=None):
    #     """Process the logits"""
    #     # if len(logits.shape) == 3:
    #     logits = P.Reshape()(logits, (-1, logits.shape[-1]))
    #     if current_index is not None:
    #         index = current_index.view((-1, 1))
    #         # print("logits before", logits.shape)
    #         # print("index", index)
    #         # logits = self.gather(logits, 0, index)
    #         index = index.to(mstype.int32)
    #         logits = P.Gather()(logits, index, 0)
    #         # print("logits after", logits.shape)
    #     logits = logits.to(mstype.float32)
    #     # print("infer logits:", logits.shape)
    #     outputs = nn.LogSoftmax()(logits)
    #     # print("infer outputs:", outputs.shape)
    #     outputs = F.tensor_pow(self.e, outputs)
    #     return outputs

    # def process_logits(self, logits, current_index=None):
    #     """Process the logits"""
    #     if current_index is not None:
    #         index = current_index.view(-1,)
    #         if len(logits.shape) == 3:
    #             logits = P.Reshape()(logits, (logits.shape[0]*logits.shape[1], -1))
    #         logits = P.Gather()(logits, index, 0)
    #     outputs = P.LogSoftmax()(logits)
    #     outputs = F.tensor_pow(self.e, outputs)
    #     return outputs

    def _forward(self,
                 origin_inputs,
                 top_k,
                 top_p,
                 repetition_penalty,
                 max_length,
                 eos_token_id):
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

        Returns:
            outputs: the ids for the generated text
        """
        # Get configurations for inference
        use_pynative = True

        batch_size = origin_inputs.shape[0]
        is_encoder_decoder = self.config.is_encoder_decoder
        # print("The input shape is: %s", origin_inputs.shape)
        valid_length_each_example = []
        target_length_each_example = []
        for i in range(batch_size):
            # As the nonzero returns the index and we need length
            valid_length = np.max(np.sum(np.not_equal(origin_inputs[i], self.config.pad_token_id), axis=-1))
            valid_length_each_example.append(valid_length)

            target_length = self.config.seq_length \
                if valid_length + max_length > self.config.seq_length \
                    else valid_length + max_length
            target_length_each_example.append(target_length)

        valid_length_each_example = np.array(valid_length_each_example)
        target_length_each_example = np.array(target_length_each_example)
        print("Get the valid for each example is: %s", valid_length_each_example)
        print("max target_length is: %s", target_length_each_example)

        # A list of the frequency of each token
        frequency_list = None
        input_ids = self._pad_inputs_using_max_length(origin_inputs=origin_inputs)

        # print("pad the origin inputs from %s into shape: %s", origin_inputs.shape, input_ids.shape)
        input_mask = np.zeros_like(input_ids)
        for i in range(valid_length_each_example.shape[0]):
            input_mask[i, :valid_length_each_example[i]] = 1

        # A single loop generates one token, loop until reaching target model_origin_max_length or generating eod token
        is_finished = [False] * batch_size
        while np.sum(is_finished) != batch_size:
            inputs = Tensor(input_ids, mstype.int32)
            seq_length = inputs.shape[1]
            current_index = [valid_length_each_example[i] - 1 + i * seq_length for i in range(batch_size)]
            # current_index = Tensor(valid_length_each_example - 1, mstype.int32)
            # print("current_index", current_index)
            current_index = Tensor(current_index, mstype.int32)
            input_mask = Tensor(input_mask, mstype.float32)
            # logits = self.model.construct(inputs, Tensor(input_mask, mstype.float32))
            output, embedding_table = self.backbone(inputs, input_mask)
            logits = self.lm_head(output, embedding_table)
            # logits = self.lm_head(output)
            log_probs = self.process_logits(logits, current_index)
            input_mask = input_mask.asnumpy()

            log_probs = log_probs.asnumpy()

            vocab_size = log_probs.shape[-1]
            if repetition_penalty != 1 and frequency_list is None:
                frequency_list = np.array([[0 for _ in range(vocab_size)]])
            log_probs_revised = log_probs.reshape(batch_size, vocab_size)
            if repetition_penalty != 1:
                log_probs_revised = log_probs - frequency_list * repetition_penalty - \
                                    (frequency_list > 0) * repetition_penalty

            p, p_args = sampler(log_probs_revised, top_p, top_k, use_pynative)
            # Random select a token as final output for this round
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                target_index = np.random.choice(len(p[i]), p=p[i])
                # update frequency list
                target = p_args[i][target_index]
                if repetition_penalty != 1:
                    frequency_list[0][target] = frequency_list[0][target] + 1
                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index]

                valid_length_each_example[i] += int(1)
                # Stop judgment
                if p_args[i][target_index] == eos_token_id or valid_length_each_example[i] == target_length_each_example[i]:
                    is_finished[i] = True
                    continue
            for i in range(batch_size):
                input_mask[i][valid_length_each_example[i] - 1] = 1
        # Return valid outputs out of padded outputs
        output_ids = []
        for i in range(batch_size):
            output_ids.append(input_ids[i, : int(valid_length_each_example[i])].astype(np.int32))
        # print("The output is: %s", output_ids)
        return output_ids

    # @ms_function
    # def model_infer(self, inputs, input_mask, current_index):
    #     output, embedding_table = self.backbone(inputs, input_mask)
    #     logits = self.lm_head(output, embedding_table)
    #     # logits = self.lm_head(output)
    #     log_probs = self.process_logits(logits, current_index)
    #     return log_probs

    def generate(self,
                 input_ids: Optional[Union[List[int], List[List[int]]]],
                 do_sample: Optional[bool] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 eos_token_id: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 max_length: Optional[int] = None):
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
        origin_phase = self.phase
        self.set_train(False)
        input_ids = np.array(input_ids).reshape(-1, np.shape(input_ids)[-1])
        config = self.config
        top_p = config.top_p if top_p is None else top_p
        top_k = config.top_k if top_k is None else top_k
        repetition_penalty = config.repetition_penalty if repetition_penalty is None else repetition_penalty
        max_length = config.max_decode_length if max_length is None else max_length
        eos_token_id = config.eos_token_id if eos_token_id is None else eos_token_id
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
                                   eos_token_id=eos_token_id)
        # set to original phase
        self.set_train(origin_phase == 'train')
        return output_ids
    
    
    ### New Feature ###

    ### New Feature ###
    def generate_graph(self,
                 input_ids,
                 top_k,
                 eos_token_id,
                 max_length):
        output_ids = self._forward_graph(input_ids,
                                         top_k,
                                         max_length,
                                         eos_token_id)
        return output_ids

    def sampler_graph(self, log_probs_revised, top_k):
        logits = F.pow(Tensor(np.e), log_probs_revised)
        # probs, p_args = ops.top_k(logits, top_k, sorted=True)
        probs, p_args = ops.TopK(sorted=True)(logits, top_k)

        norm = probs.sum(-1, keepdims=True)
        avg = F.broadcast_to(Tensor(1/top_k), probs.shape).to(mstype.float32)
        probs = F.select(norm==0, avg, probs/norm)
        return probs, p_args

    def _forward_graph(self,
                       origin_inputs,
                       top_k,
                       max_length,
                       eos_token_id):
        # print("top_k", type(top_k))
        # print("max_length", type(max_length))
        # print("eos_token_id", type(eos_token_id))
        origin_inputs = origin_inputs.to(mstype.int32)
        batch_size = origin_inputs.shape[0]

        valid_token = (origin_inputs != self.pad_token_id).to(mstype.int32)
        valid_tokens = valid_token.to(mstype.float32)
        valid_length_each_example = valid_tokens.sum(axis=1).to(mstype.int32)
        target_length_each_example = valid_length_each_example + max_length

        pad_length = self.seq_length - origin_inputs.shape[-1]
        input_ids = F.concat((origin_inputs, F.fill(mstype.int32, (batch_size, pad_length), eos_token_id)), axis=-1)
        input_mask = F.concat((valid_token, F.zeros((batch_size, pad_length), mstype.int32)), axis=-1)

        is_finished = Tensor([0.0] * batch_size,dtype=mstype.float32)
        offset = Tensor([- 1 + i * self.seq_length for i in range(batch_size)])
        while is_finished.sum() != batch_size:
            output, embedding_table = self.backbone(input_ids, input_mask.to(mstype.float32))
            # print("output:", output)
            logits = self.lm_head(output, embedding_table)
            # logits = self.lm_head(output)

            current_index = valid_length_each_example + offset
            log_probs = self.process_logits(logits, current_index)

            vocab_size = log_probs.shape[-1]
            log_probs_revised = log_probs.reshape(batch_size, vocab_size)
            p, p_args = self.sampler_graph(log_probs_revised, top_k)

            target_index = ops.multinomial(p, 1)
            for i in range(batch_size):
                if is_finished[i]:
                    continue

                input_ids[i, valid_length_each_example[i]] = p_args[i, target_index[i]].squeeze()
                valid_length_each_example[i] += 1

                if p_args[i][target_index[i]] == eos_token_id or valid_length_each_example[i] == target_length_each_example[i]:
                    is_finished[i] = 1.0
                    continue

            indices = F.concat((Tensor(np.arange(batch_size).astype(np.int32).reshape(-1, 1)), valid_length_each_example.reshape(-1,  1) - 1), axis=1)
            value = Tensor(np.ones(batch_size).astype(np.int32))
            input_mask = ops.tensor_scatter_update(input_mask, indices, value)
        return input_ids
