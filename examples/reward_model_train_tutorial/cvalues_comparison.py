import time

import numpy as np
import jsonlines
from tqdm import tqdm
from mindformers import AutoTokenizer
from mindspore.mindrecord import FileWriter
import argparse
skip_count = 0


def get_txt(tokenizer, file_path, seq_length=1024, static=True, pad_token_id=0):

    prompt_format = (
        "根据以下问题，写一个合适的回答。\n\n"
        "### 问题：\n{instruction}\n\n### 回答：\n{response}"
    )

    PAD_ID = pad_token_id

    with open(file_path, 'r', encoding='utf-8') as file:
        for item in jsonlines.Reader(file):
            sample = {}
            prompt = item["prompt"].strip()
            chosen = item["pos_resp"].strip()
            reject = item["neg_resp"].strip()
            tokenizer.pad_token_id = PAD_ID
            prompt_len = np.array(tokenizer(
                prompt,
                truncation=True,
                max_length=seq_length,
                add_special_tokens=False,
            )["input_ids"]
            ).shape[0]

            chosen_len = np.array(tokenizer(
                chosen,
                truncation=True,
                max_length=seq_length,
                add_special_tokens=False,
            )["input_ids"]
            ).shape[0]

            reject_len = np.array(tokenizer(
                reject,
                truncation=True,
                max_length=seq_length,
                add_special_tokens=False,
            )["input_ids"]
            ).shape[0]

            chosen_response_dict = tokenizer(
                prompt_format.format_map({"instruction": prompt, "response": chosen}),
                truncation=True,
                max_length=seq_length,
                padding="max_length",
                add_special_tokens=False,
            )
            rejected_response_dict = tokenizer(
                prompt_format.format_map({"instruction": prompt, "response": reject}),
                truncation=True,
                max_length=seq_length,
                padding="max_length",
                add_special_tokens=False,
            )

            sample["chosen_input_ids"] = np.array(chosen_response_dict["input_ids"], dtype=np.int32)
            sample["chosen_attention_mask"] = np.array(chosen_response_dict["attention_mask"], dtype=np.int32)
            sample["rejected_input_ids"] = np.array(rejected_response_dict["input_ids"], dtype=np.int32)
            sample["rejected_attention_mask"] = np.array(rejected_response_dict["attention_mask"], dtype=np.int32)

            try:
                divergence_idx = np.nonzero(sample["chosen_input_ids"] != sample["rejected_input_ids"])[0][0]
            except IndexError:
                skip_count += 1
                print("skip_count: ", skip_count)
                continue

            sample["position_id"] = np.arange(seq_length, dtype=np.int32)

            c_idxs = np.nonzero(sample["chosen_input_ids"] == PAD_ID)
            if len(c_idxs[0]) != 0:
                c_idx = c_idxs[0][0]
            else:
                c_idx = len(sample["chosen_input_ids"])

            r_idxs = np.nonzero(sample["rejected_input_ids"] == PAD_ID)
            if len(r_idxs[0]) != 0:
                r_idx = r_idxs[0][0]
            else:
                r_idx = len(sample["rejected_input_ids"])

            end_ind = max(c_idx, r_idx)
            loss_mask = np.zeros(seq_length, dtype=np.int32)
            loss_mask[divergence_idx:end_ind] = 1
            sample["loss_mask"] = loss_mask
            sample["end_ind"] = end_ind
            # print("prompt_len, chosen_len, reject_len", prompt_len, chosen_len, reject_len)
            yield sample, prompt_len, chosen_len, reject_len


def write_mindrecord(tokenizer, src_file, dst_file, seq_length=1024, pad_token_id=0):

    schema = {"chosen_input_ids": {"type": "int32", "shape": [-1]},
              "chosen_attention_mask": {"type": "int32", "shape": [-1]},
              "rejected_input_ids": {"type": "int32", "shape": [-1]},
              "rejected_attention_mask": {"type": "int32", "shape": [-1]},
              "position_id": {"type": "int32", "shape": [-1]},
              "loss_mask": {"type": "int32", "shape": [-1]},
              "end_ind": {"type": "int64"}, }

    writer = FileWriter(file_name=dst_file, shard_num=1, overwrite=True)
    writer.add_schema(schema)

    static_dict = {"count": 0,
                   "prompt_max": 0, "prompt_min": seq_length+1, "prompt_avg": 0,
                   "chosen_max": 0, "chosen_min": seq_length+1, "chosen_avg": 0,
                   "reject_max": 0, "reject_min": seq_length+1, "reject_avg": 0}
    prompt_total_len = 0
    chosen_total_len = 0
    reject_total_len = 0
    np.set_printoptions(threshold=np.inf)
    for item in tqdm(get_txt(tokenizer, src_file, seq_length=seq_length, pad_token_id=pad_token_id)):
        sample = item[0]
        writer.write_raw_data([sample])
        static_dict["count"] = static_dict["count"] + 1
        static_dict["prompt_max"] = item[1] if item[1] > static_dict["prompt_max"] else static_dict["prompt_max"]
        static_dict["prompt_min"] = item[1] if item[1] < static_dict["prompt_min"] else static_dict["prompt_min"]
        prompt_total_len += item[1]
        static_dict["chosen_max"] = item[2] if item[2] > static_dict["chosen_max"] else static_dict["chosen_max"]
        static_dict["chosen_min"] = item[2] if item[2] < static_dict["chosen_min"] else static_dict["chosen_min"]
        chosen_total_len += item[2]
        static_dict["reject_max"] = item[3] if item[3] > static_dict["reject_max"] else static_dict["reject_max"]
        static_dict["reject_min"] = item[3] if item[3] < static_dict["reject_min"] else static_dict["reject_min"]
        reject_total_len += item[3]

    static_dict["prompt_avg"] = prompt_total_len / static_dict["count"]
    static_dict["chosen_avg"] = chosen_total_len / static_dict["count"]
    static_dict["reject_avg"] = reject_total_len / static_dict["count"]

    print(static_dict)

    writer.commit()
    print("Transformation finished! Output file refer: {}".format(dst_file))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default="bloom_560m",
        required=True,
        help='model name for AutoTokenizer')
    parser.add_argument(
        '--padding_side',
        default="right",
        help='tokenizer padding side')
    parser.add_argument(
        '--src_file',
        default=None,
        required=True,
        help='raw data file to convert')
    parser.add_argument(
        '--dst_file',
        default=None,
        required=True,
        help='reward model data file after converting')
    parser.add_argument(
        '--seq_length',
        type=int,
        default=1024,
        required=True,
        help='sequence length of data file after converting')
    parser.add_argument(
        '--pad_token_id',
        type=int,
        default=0,
        required=True,
        help='pad_token_id')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == "__main__":
    args = get_args()
    print(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = args.padding_side
    tokenizer.pad_token = tokenizer.eos_token
    src_file = args.src_file
    dst_file = args.dst_file
    seq_length = args.seq_length
    pad_token_id = int(args.pad_token_id)
    write_mindrecord(tokenizer, src_file, dst_file, seq_length, pad_token_id)

