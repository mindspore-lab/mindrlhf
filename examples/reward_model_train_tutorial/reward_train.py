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
"""reward model Train/Eval/Predict scripts."""

import argparse
import os
import sys
import mindspore
from mindspore import context
from mindformers.core.context import build_context, build_profile_cb, init_context
from mindformers.trainer import Trainer
from mindformers.trainer.config_args import ContextConfig, ParallelContextConfig
from mindformers.tools.register.config import MindFormerConfig
from mindformers.tools.utils import str2bool
from mindformers.mindformer_book import MindFormerBook
sys.path.append(os.path.abspath('../../../'))
from mindrlhf.models.llama.llama_reward import LlamaRewardModel
from mindrlhf.models.baichuan2 import Baichuan7BReward

def run(config='run_llama_2_7b_rm.yaml',
        train_dataset='',
        run_mode='train',
        task='text_generation',
        use_parallel=True,
        resume=False):
    """Reward model training entrance."""
    if os.path.exists(config) and config.endswith(('.yaml', '.yml')):
        real_config_path = os.path.realpath(config)
        config = MindFormerConfig(real_config_path)
        model_name = config.trainer.model_name
        MindFormerBook._TRAINER_SUPPORT_TASKS_LIST[task][model_name] = real_config_path
        config.use_parallel = use_parallel
        build_context(config)
        print("config", config)
        context.set_context(jit_syntax_level=mindspore.STRICT)
        print("********************config.model_config", config.model.model_config)
    if run_mode == 'train':
        task = Trainer(args=config,
                       task=task,
                       train_dataset=train_dataset)
        task.train(train_checkpoint=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='run_internlm_7b.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--run_mode', default='train', type=str,
                        help='set run mode for model.')
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--use_parallel', default=True, type=str2bool,
                        help='open parallel for model.')
    args = parser.parse_args()
    print(args)
    run(config=args.config,
        train_dataset=args.train_dataset,
        run_mode=args.run_mode,
        task=args.task,
        use_parallel=args.use_parallel)
