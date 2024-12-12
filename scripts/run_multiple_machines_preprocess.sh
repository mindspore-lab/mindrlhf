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

MF_SCRIPTS_ROOT="$(realpath "$(dirname "$0")")"

EXECUTE_ORDER="$1"
NUM_DEVICE=$2

NUMS_NODE=$3
CUR_NODE=$4

echo "total number of machines: $NUMS_NODE, current number of machines: $CUR_NODE"

SRC=$(echo "$1" | sed -n 's/.*--src \([^ ]*\).*/\1/p')

REALPATH_MF_SCRIPTS_ROOT=$(realpath -ms "$MF_SCRIPTS_ROOT")

SRC_REALPATH_DST=$(realpath -ms --relative-to="$MF_SCRIPTS_ROOT" "$SRC")
SRC_COMBINED_PATH="$REALPATH_MF_SCRIPTS_ROOT/$SRC_REALPATH_DST"
SRC_COMBINED_PATH=$(realpath -ms "$SRC_COMBINED_PATH")

# 检查文件是否存在
if [ ! -f "$SRC_COMBINED_PATH" ]; then
    echo "Error: File $SRC_COMBINED_PATH does not exist."
    exit 1
fi

# 使用 wc -l 命令统计文件中的行数
line_count=$(wc -l < "$SRC_COMBINED_PATH")

# 计算每份文件应该包含的行数（基本行数）
lines_per_part=$(( line_count / $NUMS_NODE ))

# 计算剩余的行数（需要分配到前面的文件中）
extra_lines=$(( line_count % $NUMS_NODE ))

# 分割文件
current_line=1
for (( i=0; i<$NUMS_NODE; i++ )); do
    # 计算当前部分的行数
    if [ $i -lt $extra_lines ]; then
        part_lines=$(( lines_per_part + 1 ))
    else
        part_lines=$lines_per_part
    fi

    # 计算结束行
    end_line=$(( current_line + part_lines - 1 ))

    part_filename="${SRC%.jsonl}_$i.jsonl"
    sed -n "${current_line},${end_line}p" "$SRC_COMBINED_PATH" > "$part_filename"

    current_line=$(( end_line + 1 ))
done

EXECUTE_ORDER="$1"

UPDATE_EXECUTE_ORDER=$(echo $EXECUTE_ORDER | sed "s|--src $SRC|--src ${SRC%.jsonl}_$CUR_NODE.jsonl|")

bash $MF_SCRIPTS_ROOT/msrun_launcher.sh "$UPDATE_EXECUTE_ORDER" $NUM_DEVICE