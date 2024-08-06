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
""" Script for inference given text sample. """
import os
import json
import argparse
import jsonlines
import numpy as np

from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.train import Model
from mindspore.communication.management import get_rank
from mindspore import load_checkpoint, load_param_into_net

from mindformers import BloomRewardModel
from mindformers import AutoConfig, AutoTokenizer
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools import logger
from mindformers.tools.register import MindFormerConfig
import sys
sys.path.append(os.path.abspath('../../../'))
from mindrlhf.models.llama.llama_reward import LlamaRewardModel
from mindrlhf.models.baichuan2 import Baichuan7BReward


def read_json(file_path):
    """ Read data from file. """
    if file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            input_txt = json.load(f)
    elif file_path.endswith(".jsonl"):
        input_txt = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                input_txt.append(item)
    else:
        raise ValueError("Unsupported file format.")
    return input_txt


def run(args):
    """ Inference for given text sample. """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    input_txt = read_json(args.data_file)
    config = MindFormerConfig(args.config)

    # init context
    logger.info("..........Build Context Config..........")
    build_context(config)
    logger.info("..........Build Parallel Config..........")
    build_parallel_config(config)
    logger.info("parallel config is: %s", config.parallel_config)

    model_config = AutoConfig.from_pretrained(args.config)
    model_config.parallel_config = config.parallel_config

    if config.model.arch.type == "BloomRewardModel":
        model = BloomRewardModel(model_config)
    elif config.model.arch.type == "LlamaRewardModel":
        model = LlamaRewardModel(model_config)
    elif config.model.arch.type == "Baichuan7BReward":
        model = Baichuan7BReward(model_config)
    model.set_train(False)

    infer_model = Model(model)

    prompt_format = (
        "根据一下问题，写一个合适的回答。\n\n"
        "### 问题：\n{instruction}\n\n### 回答：\n{response}"
    )

    batch_size = 1
    seq_length = model_config.seq_length

    if args.distributed_ckpt_path is not None:
        logger.info("..........Load Distributed Checkpoints..........")
        rank_id = get_rank()
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(args.distributed_ckpt_path, "rank_{}".format(rank_id))
        ckpt_path = get_last_checkpoint(ckpt_path)
        logger.info("Checkpoint Path: %s", str(ckpt_path))

        input_ids = Tensor(np.ones(shape=(batch_size*2, seq_length)), mstype.int32)
        attention_mask = Tensor(np.ones(shape=(batch_size*2, seq_length)), mstype.float16)
        position_id = Tensor(np.ones(shape=(batch_size*2, seq_length)), mstype.int32)
        loss_mask = Tensor(np.ones(shape=(batch_size, seq_length)), mstype.float16)
        end_ind = Tensor(np.ones(shape=(batch_size*2)), mstype.int32)
        infer_model.infer_predict_layout(input_ids, position_id, attention_mask, loss_mask, end_ind)

        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        logger.info("Network parameters are not loaded: %s", str(not_load_network_params))

    for item in input_txt:
        prompt = item["prompt"].strip()
        for response in (item["pos_resp"].strip(), item["neg_resp"].strip()):
            token_dict = tokenizer(
                prompt_format.format_map({"instruction": prompt, "response": response}),
                truncation=True,
                max_length=seq_length,
                padding="max_length",
                add_special_tokens=False,
            )
            input_ids = np.expand_dims(np.array(token_dict["input_ids"]), axis=0)
            attention_mask = np.expand_dims(np.array(token_dict["attention_mask"]), axis=0)
            end_ind = attention_mask.sum(-1)
            input_ids = Tensor(input_ids, mstype.int32)
            end_ind = Tensor(end_ind, mstype.int32)
            logger.info(f"Sample:\n{prompt_format.format_map({'instruction': prompt, 'response': response})}")
            end_score = model.infer(input_ids=input_ids, end_ind=end_ind)
            logger.info(f"reward score: {end_score}")


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default="configs/bloom/run_bloom_7.1b_reward.yaml",
        required=False,
        help="YAML config file path"
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Path to test data file. Support format: [json, jsonl]"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        default="llama2_7b",
        help="Name or path of tokenizer."
    )
    parser.add_argument(
        "--distributed_ckpt_path",
        default=None,
        help="Distributed checkpoint path. When set to None, \
              load unsliced checkpoint specified in config file. Default is None."
    )
    args_, rest_args_ = parser.parse_known_args()
    if args_.config is not None:
        args_.config = os.path.join(work_path, args_.config)

    run(args_)
