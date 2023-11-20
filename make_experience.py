#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2023 Huawei Technologies Co., Ltd.All Rights Reserved.
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
# ==============================================================================
"""make experience."""
import os
import time
import math
import argparse
import mindspore
from mindspore import context
import mindspore.communication.management as D
import mindspore.nn as nn
from mindrlhf.trainer.ppo_trainer import PPOTrainer
from mindrlhf.utils.configs import init_configs, init_network_and_optimizer, init_ppo_dataset
from mindrlhf.utils.utils import set_pipeline_parallel_context

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--align_type',
        default="rlhf_stages",
        help='the name for align algorithm. Currently, It supports rlhf, rlhf_stages, dpo, dpo_stages')
    parser.add_argument(
        '--model',
        default="pangu",
        help='model name or path for align model. Currently, It supports pangu, gpt, bloom, llama')
    parser.add_argument(
        '--device_target',
        default='Ascend',
        help='device_target (str): Ascend.')
    parser.add_argument(
        '--mode',
        default=0,
        help='run mode (int): Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).')
    parser.add_argument(
        '--save_graphs',
        default=False,
        help='save_graphs (bool): True or False.')
    parser.add_argument(
        '--save_graphs_path',
        default='./graph',
        help='save_graphs_path (str): the path to save graphs.')
    parser.add_argument(
        '--enable_compile_cache',
        default=False,
        help='enable_compile_cache (bool): Whether to save or load the cache of the graph compiled by front-end')
    parser.add_argument(
        '--max_device_memory',
        default='59GB',
        help='max_device_memory (str): Set the maximum memory available for devices. The format is xxGB.')
    parser.add_argument(
        '--dataset_dir',
        default='',
        help='dataset_dir (str): dataset dir.')
    parser.add_argument(
        '--sft_model_path',
        default='/path/run_pangualpha_2_6b.yaml',
        help='sft_model_path (str): sft model yaml path.')
    parser.add_argument(
        '--critic_model_path',
        default='/path/run_pangualpha_2_6b.yaml',
        help='critic_model_path (str): critic model yaml path.')
    parser.add_argument(
        '--reward_model_path',
        default='/path/run_pangualpha_2_6b.yaml',
        help='reward_model_path (str): reward model yaml path.')
    parser.add_argument(
        '--save_data_file',
        default='/path/mindrlhf/ppodata/ppo.mindrecord',
        help='save_data_file (str): save data files.')
    args_opt = parser.parse_args()
    return args_opt

def run_rlhf(args):
    context.set_context(save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path, mode=args.mode, 
                        device_target=args.device_target, enable_compile_cache=False, 
                        compile_cache_path="./cache", max_call_depth=4096,
                        memory_optimize_level='O1', max_device_memory=args.max_device_memory)

    ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config = init_configs(args)
    set_pipeline_parallel_context(parallel_mode=ppo_config.parallel_mode, full_batch=ppo_config.full_batch,
                                  optimizer_shard=True, stage_num=sft_model_config.parallel_config.pipeline_stage,
                                  enable_alltoall=ppo_config.enable_alltoall)
    trainer = PPOTrainer(ppo_config=ppo_config, sft_model_config=sft_model_config, ref_model_config=ref_model_config,
                         critic_model_config=critic_model_config, rm_model_config=rm_model_config)
    ppo_with_grad = init_network_and_optimizer(trainer)
    rank_id = D.get_rank()
    for _ in range(ppo_config.epochs):
        # sampling
        trainer.make_experience(num_rollouts=ppo_config.num_rollouts)

    print("PPO train done!")

if __name__ == "__main__":
    args = get_args()
    run_rlhf(args)
