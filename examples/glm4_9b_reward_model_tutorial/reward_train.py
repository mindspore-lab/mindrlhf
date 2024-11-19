# Copyright 2024 Huawei Technologies Co., Ltd
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
"""reward model Train/Eval/Predict scripts."""

import argparse
import os
from mindformers.core.context import build_context
from mindformers.trainer import Trainer
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.utils import str2bool

from mindrlhf.models.glm4.glm_reward import Glm4RewardModel


def run(
    config="xxx.yaml",
    run_mode="train",
    task="text_generation",
    seq_length=None,
    train_dataset="",
    use_parallel=None,
    ckpt=None,
    strategy=None,
    auto_trans_ckpt=None,
    resume=False,
    predict_data="",
    tokenizer_path="",
):
    """Reward model training entrance."""
    assert os.path.exists(config) and config.endswith((".yaml", ".yml"))
    # init config
    config = MindFormerConfig(os.path.realpath(config))
    # variable that work in this function
    if run_mode is None:
        run_mode = config.run_mode
    if ckpt is None:
        ckpt = config.load_checkpoint

    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if use_parallel is not None:
        config.use_parallel = use_parallel
    if strategy is not None and os.path.exists(strategy):
        config.src_strategy_path_or_dir = strategy
    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
    if tokenizer_path is not None:
        config.processor.tokenizer.vocab_file = tokenizer_path

    build_context(config)

    if run_mode == "train":
        task = Trainer(args=config, task=task, train_dataset=train_dataset)
        task.train(
            train_checkpoint=ckpt,
            auto_trans_ckpt=config.auto_trans_ckpt,
            resume_training=resume,
        )
    else:
        raise NotImplementedError("run_mode only support train")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="run_glm4_9b_rm.yaml", type=str, help="set task type."
    )
    parser.add_argument(
        "--run_mode", default="train", type=str, help="set run mode for model."
    )
    parser.add_argument(
        "--task", default="text_generation", type=str, help="set task type."
    )
    parser.add_argument("--seq_length", default=None, type=int, help="seq_length")
    parser.add_argument(
        "--train_dataset", default="", type=str, help="set train dataset."
    )
    parser.add_argument(
        "--use_parallel", default=True, type=str2bool, help="open parallel for model."
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="checkpoint name or dir to load.",
    )
    parser.add_argument(
        "--src_strategy", default=None, type=str, help="strategy of load_checkpoint"
    )
    parser.add_argument(
        "--auto_trans_ckpt",
        default=None,
        type=str2bool,
        help="whether to transform checkpoint to the checkpoint matching current distribute strategy.",
    )
    parser.add_argument(
        "--resume", default=False, type=str2bool, help="whether resume training."
    )
    parser.add_argument(
        "--predict_data", default="", type=str, nargs="+", help="input predict data."
    )
    parser.add_argument(
        "--tokenizer", default=None, type=str, help="path to tokenizer model"
    )

    args = parser.parse_args()
    run(
        config=args.config,
        run_mode=args.run_mode,
        task=args.task,
        seq_length=args.seq_length,
        train_dataset=args.train_dataset,
        use_parallel=args.use_parallel,
        ckpt=args.load_checkpoint,
        strategy=args.src_strategy,
        auto_trans_ckpt=args.auto_trans_ckpt,
        resume=args.resume,
        predict_data=args.predict_data,
        tokenizer_path=args.tokenizer,
    )
