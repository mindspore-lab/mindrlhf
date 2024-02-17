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

# bash tests/st/run_distribute_test.sh /path/hccl.json 0 8 test_llama2

import os
import pytest
from collections import namedtuple

from mindspore import context

from mindrlhf.trainer.ppo_trainer import PPOTrainer
from mindrlhf.utils.configs import init_configs, init_network_and_optimizer, init_ppo_dataset
from mindrlhf.utils.utils import set_pipeline_parallel_context, get_testing_dataset_path

context.set_context(mode=context.GRAPH_MODE, max_call_depth=4096,
                    memory_optimize_level='O1', max_device_memory='25GB')

root_path = os.path.dirname(os.path.abspath(__file__)).split('tests')[0]

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_llama2_fully_inference_rlhf():
    """
    Features: Test llama2 rlhf
    Description: test llama2 rlhf
    Expectation: test pass
    """
    args = namedtuple("input_args",
                      ["dataset_dir", "sft_model_path", "reward_model_path", "critic_model_path", "save_data_file",
                       "align_type"])
    input_args = args(dataset_dir=get_testing_dataset_path("cvalues_2048"),
                      sft_model_path=f"{root_path}model_configs/llama2_config/llama2_7b.yaml",
                      reward_model_path=f"{root_path}model_configs/llama2_config/llama2_7b.yaml",
                      critic_model_path=f"{root_path}model_configs/llama2_config/llama2_7b.yaml",
                      save_data_file="",
                      align_type="")
    ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config = init_configs(input_args)
    sft_model_config.num_layers = 1
    ref_model_config.num_layers = 1
    critic_model_config.num_layers = 1
    rm_model_config.num_layers = 1
    rank_id, _ = set_pipeline_parallel_context(ppo_config)
    trainer = PPOTrainer(ppo_config=ppo_config, sft_model_config=sft_model_config, ref_model_config=ref_model_config,
                         critic_model_config=critic_model_config, rm_model_config=rm_model_config)
    ppo_with_grad = init_network_and_optimizer(trainer)
    trainer.make_experience(num_rollouts=ppo_config.num_rollouts)
    dataset = init_ppo_dataset(trainer)
    trainer.train(ppo_with_grad, dataset, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_llama2_incre_inference_rlhf():
    """
    Features: Test llama2 rlhf
    Description: test llama2 rlhf
    Expectation: test pass
    """
    args = namedtuple("input_args",
                      ["dataset_dir", "sft_model_path", "reward_model_path", "critic_model_path", "save_data_file",
                       "align_type"])
    input_args = args(dataset_dir=get_testing_dataset_path("cvalues_2048"),
                      sft_model_path=f"{root_path}model_configs/llama2_config/llama2_7b.yaml",
                      reward_model_path=f"{root_path}model_configs/llama2_config/llama2_7b.yaml",
                      critic_model_path=f"{root_path}model_configs/llama2_config/llama2_7b.yaml",
                      save_data_file="",
                      align_type="")
    ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config = init_configs(input_args)
    sft_model_config.num_layers = 1
    ref_model_config.num_layers = 1
    critic_model_config.num_layers = 1
    rm_model_config.num_layers = 1
    rank_id, _ = set_pipeline_parallel_context(ppo_config)
    trainer = PPOTrainer(ppo_config=ppo_config, sft_model_config=sft_model_config, ref_model_config=ref_model_config,
                         critic_model_config=critic_model_config, rm_model_config=rm_model_config)
    ppo_with_grad = init_network_and_optimizer(trainer)
    trainer.make_experience(num_rollouts=ppo_config.num_rollouts)
    dataset = init_ppo_dataset(trainer)
    trainer.train(ppo_with_grad, dataset, 0)
