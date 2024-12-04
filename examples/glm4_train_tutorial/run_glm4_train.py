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
"""glm4 make experience example"""

import os
import argparse

from mindspore import context
import mindspore.communication.management as D

from mindformers import MindFormerConfig, logger
from mindformers.models.glm2 import ChatGLM2Config
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config

from mindrlhf.trainer.ppo_trainer import PPOTrainer
from mindrlhf.configs.ppo_configs import PPOConfig
from mindrlhf.utils.configs import combine_config, init_ppo_dataset, init_network_and_optimizer
from mindrlhf.utils import transfer_from_str_to_bool


def main(sft_path, reward_path, critic_path, use_parallel, args):

    # init config with yaml
    sft_config = MindFormerConfig(sft_path)
    use_parallel = transfer_from_str_to_bool(use_parallel)
    args.enable_compile_cache = transfer_from_str_to_bool(args.enable_compile_cache)
    args.only_save_strategy = transfer_from_str_to_bool(args.only_save_strategy)
    sft_config.use_parallel = use_parallel
    os.environ["RUN_MODE"] = sft_config.run_mode

    # init context
    build_context(sft_config)
    build_parallel_config(sft_config)
    context.set_context(enable_compile_cache=args.enable_compile_cache, compile_cache_path='./training_cache')

    # init sft model
    sft_config.model.model_config.parallel_config = sft_config.parallel_config
    sft_config.model.model_config.use_past = False
    if args.load_sft_checkpoint == "None":
        args.load_sft_checkpoint = None
    sft_config.load_checkpoint = args.load_sft_checkpoint
    sft_model_config = ChatGLM2Config(**sft_config.model.model_config)
    sft_model_config.checkpoint_name_or_path = args.load_sft_checkpoint
    sft_model_config.model_name = "glm4"

    # init ppo config
    ppo_config = PPOConfig()
    ppo_config.mind_dataset_dir = None
    ppo_config.save_data_file = args.save_data_file
    ppo_config.save_ckpt_dir = args.save_ckpt_dir
    ppo_config.only_save_strategy = args.only_save_strategy
    ppo_config.use_parallel = use_parallel
    ppo_config.align_type = "rlhf_stages"
    ppo_config = combine_config(ppo_config, sft_model_config)

    # init ref model
    ref_config = MindFormerConfig(sft_path)
    ref_config.use_parallel = use_parallel
    ref_config.model.model_config.parallel_config = ref_config.parallel_config
    ref_config.model.model_config.use_past = False
    ref_config.load_checkpoint = None
    ref_model_config = ChatGLM2Config(**ref_config.model.model_config)
    ref_model_config.checkpoint_name_or_path = None
    ref_model_config.model_name = "glm4"

    # init reward model
    rm_config = MindFormerConfig(reward_path)
    rm_config.use_parallel = use_parallel
    rm_config.model.model_config.parallel_config = rm_config.parallel_config
    rm_config.model.model_config.use_past = False
    rm_config.load_checkpoint = None
    rm_model_config = ChatGLM2Config(**rm_config.model.model_config)
    rm_model_config.checkpoint_name_or_path = None
    rm_model_config.model_name = "glm4"

    # init critic model
    critic_config = MindFormerConfig(critic_path)
    critic_config.use_parallel = use_parallel
    critic_config.model.model_config.parallel_config = critic_config.parallel_config
    critic_config.model.model_config.use_past = False
    if args.load_critic_checkpoint == "None":
        args.load_critic_checkpoint = None
    critic_config.load_checkpoint = args.load_critic_checkpoint
    critic_model_config = ChatGLM2Config(**critic_config.model.model_config)
    critic_model_config.checkpoint_name_or_path = args.load_critic_checkpoint
    critic_model_config.model_name = "glm4"

    trainer = PPOTrainer(ppo_config=ppo_config, sft_model_config=sft_model_config, ref_model_config=ref_model_config,
                         rm_model_config=rm_model_config, critic_model_config=critic_model_config)
    trainer.pre_run("train")
    dataset = init_ppo_dataset(trainer)
    data = next(dataset.create_tuple_iterator())
    ppo_with_grad = init_network_and_optimizer(trainer)
    ppo_with_grad.set_train(True)
    # =============== pre compile ppo with grad model ===========
    ppo_with_grad.compile(*data)
    # ===========================================================
    trainer.load_checkpoint()
    trainer.train(ppo_with_grad, dataset)
    logger.info("End of training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='glm4 make experience')
    parser.add_argument('--sft_path', type=str, default=None, help='sft model path', required=True)
    parser.add_argument('--reward_path', type=str, default=None, help='reward model path', required=True)
    parser.add_argument('--critic_path', type=str, default=None, help='critic model path', required=True)
    parser.add_argument('--save_data_file', type=str, default=None, help='save_data_file', required=True)
    parser.add_argument('--save_ckpt_dir', type=str, default="./", help='save_ckpt_dir')
    parser.add_argument('--use_parallel', type=str, default=False, help='use_parallel')
    parser.add_argument('--load_sft_checkpoint', type=str, default=None, help='load checkpoint path')
    parser.add_argument('--load_critic_checkpoint', type=str, default=None, help='load checkpoint path')
    parser.add_argument('--enable_compile_cache', type=str, default=False, help='enable compile cache')
    parser.add_argument('--only_save_strategy', type=str, default=False, help='only save strategy')
    args = parser.parse_args()
    main(args.sft_path, args.reward_path, args.critic_path, args.use_parallel, args)

