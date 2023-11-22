import os
from transformers import AutoTokenizer
from datasets import load_dataset
from mindformers import GPT2Tokenizer

import numpy as np
from mindspore.mindrecord import FileWriter

MR_PATH = "TLDR_data_ms"


def writeMR(samples, original_samples, file_name):
    mr_list = []
    batch_size = 1
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=500,
            padding="max_length",
            return_tensors="pt",
        )
        # input_ids = encodings_dict["input_ids"].to(rw_device)
        # attn_masks = encodings_dict["attention_mask"].to(rw_device)

        # mr_list.append({"input_ids":input_ids.numpy(),
        #                 "attention_mask":attn_masks.numpy()})
        prompt_ids = encodings_dict["input_ids"].to(rw_device)
        prompt_mask = encodings_dict["attention_mask"].to(rw_device)

        sub_samples_2 = original_samples[i: i + batch_size]
        sub_samples_2 = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples_2]
        encodings_dict_2 = rw_tokenizer(
            sub_samples_2,
            truncation=True,
            max_length=550,
            padding="max_length",
            return_tensors="pt",
        )

        original_sample_ids = encodings_dict_2["input_ids"].to(rw_device)
        original_sample_mask = encodings_dict_2["attention_mask"].to(rw_device)

        mr_list.append({"prompt_ids": prompt_ids.numpy(),
                        "prompt_mask": prompt_mask.numpy(),
                        "original_sample_ids": original_sample_ids.numpy(),
                        "original_sample_mask": original_sample_mask.numpy()})

    # define columns
    nlp_schema = {
        "prompt_ids": {"type": "int64", "shape": [-1]},
        "prompt_mask": {"type": "int64", "shape": [-1]},
        "original_sample_ids": {"type": "int64", "shape": [-1]},
        "original_sample_mask": {"type": "int64", "shape": [-1]},
    }

    mr_writer = FileWriter(file_name, shard_num=1, overwrite=True)
    mr_writer.add_schema(nlp_schema, "Transfered trlx train dataset.")

    if mr_list:
        mr_writer.write_raw_data(mr_list)
    mr_writer.commit()


def ms_writeMR(samples, file_name):
    mr_list = []
    batch_size = 1
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i: i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen for chosen in sub_samples]
        encodings_dict = ms_tokenizer(
            sub_samples,
            truncation=True,
            max_length=553,
            padding="max_length",
            # add_special_tokens=False,
            return_tensors="ms",
        )
        input_ids = encodings_dict["input_ids"]
        if i == 0:
            print(input_ids.shape)
            print(input_ids)
        attn_masks = encodings_dict["attention_mask"]
        mr_list.append({"input_ids": input_ids.numpy()[:, 1:-2],
                        "attention_mask": attn_masks.numpy()[:, 1:-2]})
    # save as mindrecord file
    # define columns
    nlp_schema = {
        "input_ids": {"type": "int64", "shape": [-1]},
        "attention_mask": {"type": "int64", "shape": [-1]},
    }

    mr_writer = FileWriter(file_name, shard_num=1, overwrite=True)
    mr_writer.add_schema(nlp_schema, "Transfered trlx train dataset.")

    if mr_list:
        mr_writer.write_raw_data(mr_list)
    mr_writer.commit()


def get_prompt_dataset(prompts):
    """
    Get the prompt after T5 decoding to make sure dictionary
    of prompts and summaries is consistent decode prompt from trlX pipeline
    """
    formatted_prompts = []
    for i in range(len(prompts)):
        tmp = ms_tokenizer.decode(
            tokenizer(
                prompts[i].split("TL;DR:")[0],
                truncation=True,
                max_length=493,  # to make sure "TL;DR" dont get truncated
                add_special_tokens=False,
            )["input_ids"],
            skip_special_tokens=True,
        ).strip()
        tmp = tmp + "\nTL;DR:"
        tmp = ms_tokenizer.decode(
            tokenizer(tmp, truncation=True, max_length=500, add_special_tokens=False)["input_ids"],
            skip_special_tokens=True,
        ).strip()
        formatted_prompts.append(tmp)
    return formatted_prompts


def ms_get_prompt_dataset(prompts):
    """
    Get the prompt after T5 decoding to make sure dictionary
    of prompts and summaries is consistent decode prompt from trlX pipeline
    """
    formatted_prompts = []
    for i in range(len(prompts)):
        tmp = ms_tokenizer.decode(
            ms_tokenizer(
                prompts[i].split("TL;DR:")[0],
                truncation=True,
                max_length=493,  # to make sure "TL;DR" dont get truncated
                padding="max_length",
                add_special_tokens=False,
            )["input_ids"],
            skip_special_tokens=True,
        ).strip()
        tmp = tmp + "\nTL;DR:"
        tmp = ms_tokenizer.decode(
            ms_tokenizer(
                tmp,
                truncation=True,
                max_length=500,
                padding="max_length",
                add_special_tokens=False
            )["input_ids"],
            skip_special_tokens=True,
        ).strip()
        formatted_prompts.append(tmp)
    return formatted_prompts


if __name__ == '__main__':
    # iniitialize tokenizer
    # uncomment when using transformers tokenizer
    rw_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_device = 'cpu'  # torch.device("cuda:{}".format(1))  # set reward model device
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # do not comment these lines
    ms_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ms_tokenizer.padding = 'max_length'
    ms_tokenizer.padding_side = "left"

    # download TLDR dataset
    dataset = load_dataset("CarperAI/openai_summarize_tldr")
    train_set = [(sample["prompt"], sample["label"]) for sample in dataset["train"]]
    val_set = [(sample["prompt"], sample["label"]) for sample in dataset["valid"]]
    train_posts, train_summaries = zip(*train_set)
    val_posts, val_summaries = zip(*val_set)

    # process train set
    post_summary_dict = {}

    # uncomment when using transformers tokenizer
    train_prompts = get_prompt_dataset(train_posts)
    # uncomment when using mindspore tokenizer
    # train_prompts = ms_get_prompt_dataset(train_posts)

    for i in range(len(train_prompts)):
        post_summary_dict[train_prompts[i]] = train_summaries[i]

    original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in train_prompts]
    original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
    if not os.path.exists(os.path.join(MR_PATH, "train")):
        os.makedirs(os.path.join(MR_PATH, "train"))
    # uncomment when using transformers tokenizer
    original_scores = writeMR(train_prompts, original_samples, "TLDR_data/train/tldr_train_prompts.mindrecord")
    # uncomment when using mindspore tokenizer
    # original_scores = ms_writeMR(original_samples, "TLDR_data_ms/train/tldr_train.mindrecord")

    # process validation set
    original_samples = []
    post_summary_dict.clear()
    original_samples.clear()

    # uncomment when using transformers tokenizer
    val_prompts = get_prompt_dataset(val_posts)
    # uncomment when using mindspore tokenizer
    # val_prompts = ms_get_prompt_dataset(val_posts)

    for i in range(len(val_prompts)):
        post_summary_dict[val_prompts[i]] = val_summaries[i]

    original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in val_prompts]
    original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]

    if not os.path.exists(os.path.join(MR_PATH, "val")):
        os.makedirs(os.path.join(MR_PATH, "val"))
    # uncomment when using transformers tokenizer
    original_scores = writeMR(val_prompts, original_samples, "TLDR_data/val/tldr_val_prompts.mindrecord")
    # uncomment when using mindspore tokenizer
    # original_scores = ms_writeMR(val_prompts, "TLDR_data_ms/val/tldr_val.mindrecord")
