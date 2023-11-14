#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
echo "bash run_distributed_inference.sh DATA_DIR RANK_TABLE_FILE RANK_SIZE RANK_START LOCAL_DEVICE_NUM"
echo "for example:"
echo "bash run_distributed_train.sh /path/dataset /path/hccl.json 8 0 8"
echo "It is better to use absolute path."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
DATA_DIR=$1
export RANK_TABLE_FILE=$2
RANK_SIZE=$3

RANK_START=$4
LOCAL_DEVICE_NUM=${5}

for((i=0;i<${LOCAL_DEVICE_NUM};i++));
do
    rm ${ROOT_PATH}/device$i/ -rf
    mkdir ${ROOT_PATH}/device$i
    cd ${ROOT_PATH}/device$i || exit
    export RANK_ID=$[i+RANK_START]
    export DEVICE_ID=$i
    python3 ${ROOT_PATH}/test_actor_inference.py --distribute=true --device_num=$RANK_SIZE --data_url=$DATA_DIR --run_type=train  > log$i.log 2>&1 &
done
