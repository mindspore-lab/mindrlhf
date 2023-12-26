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
echo "bash run_distributed_train.sh EXECUTE_ORDER RANK_TABLE_PATH DEVICE_RANGE RANK_SIZE"
echo ""
echo "for example:"
echo "bash run_distribute_reward.sh 'python reward_train.py --run_mode=train --use_parallel True --config run_llama_2_7b_rm.yaml \
    --train_dataset train_4096.mindrecord' hccl_4p_0123_127.0.0.1.json [0,4] 4"
echo "It is better to use absolute path."
echo "=============================================================================================================="

check_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

EXECUTE_ORDER=$1
RANK_TABLE_PATH=$(check_real_path $2)
DEVICE_RANGE=$3

DEVICE_RANGE_LEN=${#DEVICE_RANGE}
DEVICE_RANGE=${DEVICE_RANGE:1:DEVICE_RANGE_LEN-2}
PREFIX=${DEVICE_RANGE%%","*}
INDEX=${#PREFIX}
START_DEVICE=${DEVICE_RANGE:0:INDEX}
END_DEVICE=${DEVICE_RANGE:INDEX+1:DEVICE_RANGE_LEN-INDEX}

ulimit -u unlimited
export RANK_SIZE=$4
export RANK_TABLE_FILE=$RANK_TABLE_PATH

shopt -s extglob
for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((i-START_DEVICE))
    mkdir -p ./output/log/rank_$RANK_ID
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    $EXECUTE_ORDER &> ./output/log/rank_$RANK_ID/mindformer.log &
done
wait
shopt -u extglob
