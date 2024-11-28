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
SAVE_DATA_FILE=$4
DATASET_FILE=$5
ONLY_SAVE_STRATEGY=${6:-False}
SAVE_CKPT_DIR=${7:-'./'}
COMPILE_CACHE=${8:-False}
DEV_NUM=${9:-1}
USE_PARALLEL=${10:-False}
SFT_CKPT=${11:-None}
RM_CKPT=${12:-None}
CRITIC_CKPT=${13:-None}
REF_CKPT=${14:-None}



script_path="$(realpath "$(dirname "$0")")"

msrun --worker_num=$DEV_NUM --local_worker_num=$DEV_NUM \
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