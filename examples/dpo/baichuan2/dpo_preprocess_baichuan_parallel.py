import argparse
import json

import numpy as np
from tqdm import tqdm
import json
from mindspore.mindrecord import FileWriter

import mindspore as ms
from mindformers import AutoModel
from mindrlhf.models.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer

from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.core.parallel_config import build_parallel_config
from mindformers.tools import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.core.context import build_context
from mindspore import ops as P
from mindspore.communication.management import get_rank
import mindspore.communication.management as D

ROLE_MAPPING = {
    "human": "<|user|>",
    "gpt": "<|assistant|>",
    "system": "<|system|>"
}


def build_message(tokenizer, messages, metadata=""):
    encoded_messages = []
    for i, msg in enumerate(messages):
        role = ROLE_MAPPING.get(msg['from'], "")
        if not role:
            raise ValueError(f"Unsupported role {msg['from']}")
        message = f"{role}{metadata}\n{msg['value']}"
        tokens = tokenizer.encode(message)
        if i != 0:
            tokens = tokens[2:]  # remove prefix
        encoded_messages.append(tokens)
    prompt_ids = []
    for encoded_ids in encoded_messages[:-1]:
        prompt_ids += encoded_ids
    answer_ids = encoded_messages[-1]
    return prompt_ids, answer_ids


def build_message_cvalues(tokenizer, prompt, ans, metadata=""):
    msg = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer.encode(msg)
    msg = f"{ans}<|im_end|>"
    answer_ids = tokenizer.encode(msg)
    return prompt_ids, answer_ids


def get_logps(model, input_ids, labels, attention_mask, loss_mask):
    """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, seq_len, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with value of label_pad_token_id are ignored. Shape: (batch_size, seq_len)

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
    valid_length = np.array(attention_mask).sum(axis=-1)
    batch_length = int(max(valid_length))
    input_ids = ms.Tensor(input_ids, dtype=ms.int32)
    if len(input_ids.shape) == 1:
        input_ids = ms.ops.unsqueeze(input_ids, 0)
    input_ids = P.StridedSlice()(input_ids, (0, 0), (input_ids.shape[0], min(batch_length, input_ids.shape[1] - 1)),
                                 (1, 1))
    labels = ms.Tensor(labels, dtype=ms.int32)
    if len(labels.shape) == 1:
        labels = ms.ops.unsqueeze(labels, 0)
    labels = P.StridedSlice()(labels, (0, 0), (labels.shape[0], min(batch_length, labels.shape[1] - 1)), (1, 1))
    loss_mask = ms.Tensor(loss_mask, dtype=ms.int32)
    if len(loss_mask.shape) == 1:
        loss_mask = ms.ops.unsqueeze(loss_mask, 0)
    loss_mask = P.StridedSlice()(loss_mask, (0, 0), (loss_mask.shape[0], min(batch_length, loss_mask.shape[1] - 1)),
                                 (1, 1))
    outputs = model(input_ids)
    logits = outputs[0]
    # [bs, seq_len] -> [bs, seq_len]
    labels = labels * loss_mask
    logits = logits.to(ms.float32)
    # [bs, seq_len, vocab_size]
    log_probs = ms.ops.log_softmax(logits, -1)
    # [bs, seq_len] -> [bs, seq_len, 1]
    index = ms.ops.unsqueeze(labels, -1)
    index = index.to(ms.int32)
    # [bs, seq_len, 1]
    per_token_logps = ms.ops.gather_elements(log_probs, -1, index)
    # [bs, seq_len, 1] -> [bs, seq_len]
    per_token_logps = ms.ops.squeeze(per_token_logps, -1)
    logps = ms.ops.sum(per_token_logps * loss_mask, -1)
    return logps.asnumpy()


def preprocess(data_path: str, dst_file: str, config_path: str,
               tokenizer_path: str, seq_len: int, dataset_type: str = 'dpo'):
    config = MindFormerConfig(config_path)
    logger.info("..........Build Context Config..........")
    print('config', config)
    build_context(config)
    logger.info("..........Build Parallel Config..........")
    build_parallel_config(config)
    logger.info("parallel config is: %s", config.parallel_config)
    rank_id = config.local_rank
    tokenizer = Baichuan2Tokenizer(vocab_file=tokenizer_path)
    model = AutoModel.from_config(config)
    model.set_train(False)
    dynamic_input_ids = ms.Tensor(shape=[None, None], dtype=ms.int32)
    model.set_inputs(dynamic_input_ids)
    if dataset_type == 'dpo':
        with open(data_path, "r", encoding='utf-8') as file:
            pairs = json.load(file)
    elif dataset_type == 'cvalues':
        pairs = []
        with open(data_path, "r", encoding='utf-8') as file:
            for line in file:
                pairs.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    schema = {
        "chosen_input_ids": {"type": "int32", "shape": [-1]},
        "chosen_labels": {"type": "int32", "shape": [-1]},
        "chosen_attention_mask": {"type": "int32", "shape": [-1]},
        "chosen_loss_mask": {"type": "int32", "shape": [-1]},
        "chosen_ref_logps": {"type": "float32", "shape": [-1]},
        "rejected_input_ids": {"type": "int32", "shape": [-1]},
        "rejected_labels": {"type": "int32", "shape": [-1]},
        "rejected_attention_mask": {"type": "int32", "shape": [-1]},
        "rejected_loss_mask": {"type": "int32", "shape": [-1]},
        "rejected_ref_logps": {"type": "float32", "shape": [-1]},
    }
    if rank_id == 0:
        writer = FileWriter(file_name=dst_file, shard_num=1, overwrite=True)
        writer.add_schema(schema)

    batch_chosen_input_ids = []
    batch_chosen_labels = []
    batch_chosen_attention_mask = []
    batch_chosen_loss_mask = []
    batch_rejected_input_ids = []
    batch_rejected_labels = []
    batch_rejected_attention_mask = []
    batch_rejected_loss_mask = []
    for pair in tqdm(pairs):
        if dataset_type == 'dpo':
            chosen_messages = pair['conversations'] + [pair['chosen']]
            rejected_messages = pair['conversations'] + [pair['rejected']]
            prompt_ids, chosen_ids = build_message(tokenizer, chosen_messages)
            _, rejected_ids = build_message(tokenizer, rejected_messages)
        elif dataset_type == 'cvalues':
            prompt_ids, chosen_ids = build_message_cvalues(tokenizer, pair['prompt'], pair['pos_resp'])
            _, rejected_ids = build_message_cvalues(tokenizer, pair['prompt'], pair['neg_resp'])
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        def _build(prompt_ids, resp_ids):
            # check input_ids > seq_length
            input_ids = prompt_ids + resp_ids
            labels = input_ids[1:] + [tokenizer.pad_token_id]
            attention_mask = [1] * len(input_ids)
            loss_mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
            input_len = len(input_ids)
            input_ids = input_ids + [0] * (seq_len - input_len)
            labels = labels + [0] * (seq_len - input_len)
            attention_mask = attention_mask + [0] * (seq_len - input_len)
            loss_mask = loss_mask + [0] * (seq_len - input_len)

            if len(input_ids) > seq_len:
                input_ids = input_ids[:seq_len]
                labels = labels[:seq_len]
                attention_mask = attention_mask[:labels]
                loss_mask = loss_mask[:seq_len]

            input_ids = np.array(input_ids, dtype=np.int32)
            labels = np.array(labels, dtype=np.int32)
            attention_mask = np.array(attention_mask, dtype=np.int32)
            loss_mask = np.array(loss_mask, dtype=np.int32)
            return input_ids, labels, attention_mask, loss_mask

        chosen_input_ids, chosen_labels, chosen_attention_mask, chosen_loss_mask = \
            _build(prompt_ids, chosen_ids)
        rejected_input_ids, rejected_labels, rejected_attention_mask, rejected_loss_mask = \
            _build(prompt_ids, rejected_ids)
        if len(batch_chosen_input_ids) == config.model.model_config.batch_size:
            batch_chosen_ref_logps = get_logps(model, batch_chosen_input_ids, batch_chosen_labels,
                                               batch_chosen_attention_mask, batch_chosen_loss_mask)
            batch_rejected_ref_logps = get_logps(model, batch_rejected_input_ids, batch_rejected_labels,
                                                 batch_rejected_attention_mask, batch_rejected_loss_mask)
            for i in range(config.model.model_config.batch_size):
                sample = {
                    "chosen_input_ids": batch_chosen_input_ids[i],
                    "chosen_labels": batch_chosen_labels[i],
                    "chosen_attention_mask": batch_chosen_attention_mask[i],
                    "chosen_loss_mask": batch_chosen_loss_mask[i],
                    "chosen_ref_logps": np.array([batch_chosen_ref_logps[i]]),
                    "rejected_input_ids": batch_rejected_input_ids[i],
                    "rejected_labels": batch_rejected_labels[i],
                    "rejected_attention_mask": batch_rejected_attention_mask[i],
                    "rejected_loss_mask": batch_rejected_loss_mask[i],
                    "rejected_ref_logps": np.array([batch_rejected_ref_logps[i]]),
                }

                if rank_id == 0:
                    writer.write_raw_data([sample])
            batch_chosen_input_ids = []
            batch_chosen_labels = []
            batch_chosen_attention_mask = []
            batch_chosen_loss_mask = []
            batch_rejected_input_ids = []
            batch_rejected_labels = []
            batch_rejected_attention_mask = []
            batch_rejected_loss_mask = []

        batch_chosen_input_ids.append(chosen_input_ids)
        batch_chosen_labels.append(chosen_labels)
        batch_chosen_attention_mask.append(chosen_attention_mask)
        batch_chosen_loss_mask.append(chosen_loss_mask)
        batch_rejected_input_ids.append(rejected_input_ids)
        batch_rejected_labels.append(rejected_labels)
        batch_rejected_attention_mask.append(rejected_attention_mask)
        batch_rejected_loss_mask.append(rejected_loss_mask)

    # last batch
    if len(batch_chosen_input_ids) == config.model.model_config.batch_size:
        batch_chosen_ref_logps = get_logps(model, batch_chosen_input_ids, batch_chosen_labels,
                                           batch_chosen_attention_mask, batch_chosen_loss_mask)
        batch_rejected_ref_logps = get_logps(model, batch_rejected_input_ids, batch_rejected_labels,
                                             batch_rejected_attention_mask, batch_rejected_loss_mask)
        for i in range(config.model.model_config.batch_size):
            sample = {
                "chosen_input_ids": batch_chosen_input_ids[i],
                "chosen_labels": batch_chosen_labels[i],
                "chosen_attention_mask": batch_chosen_attention_mask[i],
                "chosen_loss_mask": batch_chosen_loss_mask[i],
                "chosen_ref_logps": np.array([batch_chosen_ref_logps[i]]),
                "rejected_input_ids": batch_rejected_input_ids[i],
                "rejected_labels": batch_rejected_labels[i],
                "rejected_attention_mask": batch_rejected_attention_mask[i],
                "rejected_loss_mask": batch_rejected_loss_mask[i],
                "rejected_ref_logps": np.array([batch_rejected_ref_logps[i]]),
            }

            if rank_id == 0:
                writer.write_raw_data([sample])
    if rank_id == 0:
        writer.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Path to source json file.")
    parser.add_argument("--dst", type=str, help="Path to target mindrecrod file.")
    parser.add_argument("--config", type=str, help="Path to model config file.")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer model file.")
    parser.add_argument('--seq_len', default=1024, type=int, help="Sequence length.")
    parser.add_argument('--dataset_type', type=str, default='dpo', help="Dataset type to process.")
    args = parser.parse_args()
    preprocess(args.src, args.dst, args.config, args.tokenizer, args.seq_len, args.dataset_type)


if __name__ == "__main__":
    main()
