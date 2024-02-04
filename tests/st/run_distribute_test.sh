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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distributed_test.sh RANK_TABLE_FILE RANK_START LOCAL_DEVICE_NUM ST_NAME"
echo "for example:"
echo "#######no pipeline#######"
echo "bash run_distributed_test.sh /path/hccl.json 0 1 test_gpt2"
echo "It is better to use absolute path."
echo "=============================================================================================================="

ROOT_PATH=`pwd`
export RANK_TABLE_FILE=$1
RANK_START=$2
LOCAL_DEVICE_NUM=$3
ST_NAME=$4

# Create st log file
rm ${ROOT_PATH}/tests/st/${ST_NAME}_log -rf
mkdir ${ROOT_PATH}/tests/st/${ST_NAME}_log

for((i=0;i<${LOCAL_DEVICE_NUM};i++));
do
    mkdir ${ROOT_PATH}/tests/st/${ST_NAME}_log/device$[i+RANK_START]
    cd ${ROOT_PATH}/tests/st/${ST_NAME}_log/device$[i+RANK_START] || exit
    export RANK_ID=$i
    export DEVICE_ID=$[i+RANK_START]
    pytest -s ${ROOT_PATH}/tests/st/${ST_NAME}.py > ${ROOT_PATH}/tests/st/${ST_NAME}_log/device$[i+RANK_START]/log$[i+RANK_START].log 2>&1 &
done
