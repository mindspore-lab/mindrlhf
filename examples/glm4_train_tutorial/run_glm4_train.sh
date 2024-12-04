#!/bin/bash
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
T_SFT_PATH=$1
T_RM_PATH=$2
T_CRITIC_PATH=$3
SAVE_DATA_FILE=$4
ONLY_SAVE_STRATEGY=${5:-False}
SAVE_CKPT_DIR=${6:-'./'}
COMPILE_CACHE=${7:-False}
DEV_NUM=${8:-1}
USE_PARALLEL=${9:-False}
SFT_CKPT=${10:-None}
RM_CKPT=${11:-None}
CRITIC_CKPT=${12:-None}
REF_CKPT=${13:-None}

script_path="$(realpath "$(dirname "$0")")"

msrun --worker_num=$DEV_NUM --local_worker_num=$DEV_NUM \
      --master_addr=127.0.0.1 --master_port=10969 \
      --join=True --log_dir=./glm_train_log \
      "$script_path"/run_glm4_train.py \
      --sft_path "$T_SFT_PATH" \
      --reward_path "$T_RM_PATH" \
      --critic_path "$T_CRITIC_PATH" \
      --save_data_file "$SAVE_DATA_FILE" \
      --save_ckpt_dir "$SAVE_CKPT_DIR" \
      --only_save_strategy "$ONLY_SAVE_STRATEGY" \
      --enable_compile_cache "$COMPILE_CACHE" \
      --use_parallel "$USE_PARALLEL" \
      --load_sft_checkpoint "$SFT_CKPT" \
      --load_critic_checkpoint "$CRITIC_CKPT"