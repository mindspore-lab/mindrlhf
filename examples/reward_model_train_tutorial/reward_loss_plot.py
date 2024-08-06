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
"""parse loss from log and converte to events.out.tfevents file that tensorboard can read"""

import argparse
import re
import os
import shutil
import time
import sys
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator


def parse_line(text):
    loss = re.findall(r"loss: (\d+\.\d+)", text)
    if loss:
        return float(loss[0])
    else:
        return None


def parse_and_convert(log_file, loss_dir, sink_size, parse_mode):
    print("start parse...")
    if os.path.exists(loss_dir):
        shutil.rmtree(loss_dir)

    count = 0
    writer = SummaryWriter(loss_dir)
    with open(log_file, 'r', encoding='UTF-8') as f:
        # Real-time monitoring and conversion
        if parse_mode == "real-time":
            while True:
                last_line = f.tell()
                line = f.readline()
                if not line:
                    print("pause parsing..., size=", count)
                    time.sleep(60)
                    f.seek(last_line)
                else:
                    loss = parse_line(line)
                    if loss:
                        count += sink_size
                        writer.add_scalar("loss", loss, count)
                        print(f"Real-time parsing, step:{count} loss:{loss} ")
        # One-time conversion after training
        else:
            lines = f.readlines()
            for line in lines:
                loss = parse_line(line)
                if loss:
                    count += sink_size
                    writer.add_scalar("loss", loss, count)
                    print(f"One-time parsing, step:{count} loss:{loss} ")

    print("end parse..., size=", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='output/log/rank_0/info.log', type=str,
                        help='the path of info.log.')
    parser.add_argument('--output_file_dir', default='loss_dir', type=str,
                        help='the directory of generated tfevents file.')
    parser.add_argument('--sink_size', default='2', type=int,
                        help='the sink_size of model config.')
    parser.add_argument('--parse_mode', default='real-time', type=str,
                        help='set the mode for parsing logfiles, real-time or one-time.')
    args = parser.parse_args()
    parse_and_convert(args.log_file, args.output_file_dir, args.sink_size, args.parse_mode)
