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
""" Script for calculating accuracy on test dataset. """
import os
import argparse
import numpy as np

from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.train import Model
from mindspore.communication.management import get_rank
from mindspore import load_checkpoint, load_param_into_net

from mindformers import BloomRewardModel
from mindformers import AutoConfig
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools import logger
from mindformers.tools.register import MindFormerConfig
from mindformers.dataset import RewardModelDataset
from mindformers.dataset import check_dataset_config


def run(args):
    """ calculate accuracy on test dataset """

    config = MindFormerConfig(args.config)

    # init context
    logger.info("..........Build Context Config..........")
    build_context(config)
    logger.info("..........Build Parallel Config..........")
    build_parallel_config(config)
    logger.info("parallel config is: %s", config.parallel_config)

    model_config = AutoConfig.from_pretrained(args.config)
    model_config.parallel_config = config.parallel_config

    # init model
    model = BloomRewardModel(model_config)
    model.set_train(False)

    infer_model = Model(model)

    batch_size = model_config.batch_size if model_config.batch_size else 1
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

    check_dataset_config(config)
    val_dataset = RewardModelDataset(config.eval_dataset_task.dataset_config)

    count = 0
    total_acc = 0
    for data in val_dataset.create_dict_iterator():
        end_scores = model.infer(data["input_ids"], data["attention_mask"])
        end_scores = end_scores.asnumpy()
        bs = end_scores.shape[0] // 2
        chosen_end_scores = end_scores[:bs]
        reject_end_scores = end_scores[bs:]
        accuracy = np.sum(chosen_end_scores > reject_end_scores) / bs
        count += 1
        total_acc += accuracy
        print(f"acc: [{accuracy}]; avg acc: [{total_acc / count}]", flush=True)
    print(f"acc: [{total_acc / count}]", flush=True)


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
        "--tokenizer",
        required=True,
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
