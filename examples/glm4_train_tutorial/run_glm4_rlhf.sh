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
G_SFT_PATH=$1
G_RM_PATH=$2
G_CRITIC_PATH=$3
T_SFT_PATH=$4
T_RM_PATH=$5
T_CRITIC_PATH=$6
SAVE_DATA_FILE=$7
DATASET_FILE=$8
ONLY_SAVE_STRATEGY=${9:-False}
SAVE_CKPT_DIR=${10:-'./'}
COMPILE_CACHE=${11:-False}
DEV_NUM=${12:-1}
USE_PARALLEL=${13:-False}
SFT_CKPT=${14:-None}
RM_CKPT=${15:-None}
CRITIC_CKPT=${16:-None}
REF_CKPT=${17:-None}


script_path="$(realpath "$(dirname "$0")")"
epoch=0

while [ $epoch -lt 10 ]; do
  echo "Epoch number is $epoch"
  epoch=$((epoch+1))
  msrun --worker_num=2 --local_worker_num=2 \
        --master_addr=127.0.0.1 --master_port=10969 \
        --join=True --log_dir=./glm_generate_log \
        "$script_path"/run_glm4_generate.py \
         --sft_path "$G_SFT_PATH" \
         --reward_path "$G_RM_PATH" \
         --critic_path "$G_CRITIC_PATH" \
         --save_data_file "$SAVE_DATA_FILE" \
         --mind_dataset_dir "$DATASET_FILE" \
         --only_save_strategy "$ONLY_SAVE_STRATEGY" \
         --enable_compile_cache "$COMPILE_CACHE" \
         --use_parallel "$USE_PARALLEL" \
         --load_sft_checkpoint "$SFT_CKPT" \
         --load_rm_checkpoint "$RM_CKPT" \
         --load_critic_checkpoint "$CRITIC_CKPT" \
         --load_ref_checkpoint "$REF_CKPT"

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
done
