#!/bin/bash
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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_train.sh DATA_DIR RANK_TABLE_FILE RANK_START LOCAL_DEVICE_NUM SFT_MODEL_PATH, REWARD_MODEL_PATH"
echo "for example:"
echo "#######no pipeline#######"
echo "bash run_distributed_train.sh /path/train.mindrecord /path/hccl.json 0 8 /path/sft_model /path/reward_model"
echo "It is better to use absolute path."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
echo $ROOT_PATH
DATA_DIR=$1
export RANK_TABLE_FILE=$2

RANK_START=$3
LOCAL_DEVICE_NUM=${4}
SFT_MODEL_PATH=$5
REWARD_MODEL_PATH=$6

for((i=${RANK_START};i<(${LOCAL_DEVICE_NUM}+${RANK_START});i++));
do
    rm ${ROOT_PATH}/device$[i]/ -rf
    mkdir ${ROOT_PATH}/device$[i]
    cd ${ROOT_PATH}/device$[i] || exit
    export RANK_ID=$[i-RANK_START]
    export DEVICE_ID=$i
    python3 ${ROOT_PATH}/train.py --dataset_dir ${DATA_DIR} --sft_model_path ${SFT_MODEL_PATH} \
  --critic_model_path ${REWARD_MODEL_PATH} --reward_model_path ${REWARD_MODEL_PATH} > log$[i].log 2>&1 &
done
