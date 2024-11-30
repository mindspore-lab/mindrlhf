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

import os
import pytest
from mindrlhf.models.qwen2.qwen_dpo import Qwen7BDPO
from mindrlhf.models.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from mindformers.tools.download_tools import download_with_progress_bar

root_path = os.path.dirname(os.path.abspath(__file__)).split('tests')[0]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend91setup_cmd0b_training
@pytest.mark.env_onecard
class TestQwen2DPO:
    @staticmethod
    def setup_cmd(scripts_cmd, device_nums):
        cmd = f"msrun --worker_num={device_nums} " + \
              f"--local_worker_num={device_nums} " + \
              f"--master_port=8118 " + \
              f"--log_dir=msrun_log " + \
              f"--join=True " + \
              f"--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        return cmd

    @pytest.mark.run(order=1)
    def test_qwen2_dpo_process(self):
        download_with_progress_bar("https://www.modelscope.cn/models/Qwen/Qwen2-7B/resolve/master/vocab.json",
                                   f"{root_path}/checkpoint_download/qwen2/vocab.json")
        download_with_progress_bar("https://www.modelscope.cn/models/Qwen/Qwen2-7B/resolve/master/merges.txt",
                                   f"{root_path}/checkpoint_download/qwen2/merges.txt")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/mindrlhf/tools/dpo_preprocess.py"

        scripts_cmd = f"{scripts_path} --src={root_path}/datasets/cvalues/source/one.jsonl " + \
                      f"--dst={root_path}/datasets/cvalues/source/qwen.mindrecord " + \
                      f"--config={root_path}/model_configs/qwen_config/process_qwen2_7b.yaml " + \
                      f"--tokenizer={root_path}/checkpoint_download/qwen2/vocab.json " + \
                      f"--merges_file={root_path}/checkpoint_download/qwen2/merges.txt " + \
                      f"--seq_len=4097 " + \
                      f"--dataset_type=cvalues " + \
                      f"--save_interval=2"
        ret = os.system(self.setup_cmd(scripts_cmd, 8))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
        os.system(f"python {root_path}/mindrlhf/tools/dpo_preprocess.py \
                    --merge True --src={root_path}/datasets/cvalues/source/ \
                    --dst {root_path}/datasets/cvalues/source/qwen.mindrecord")

        assert os.path.isfile(f"{root_path}/datasets/cvalues/source/qwen.mindrecord")

    @pytest.mark.run(order=2)
    def test_qwen2_finetune(self):
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/run_dpo.py"

        scripts_cmd = f"{scripts_path} --config={root_path}/model_configs/qwen_config/finetune_qwen2_7b_dpo.yaml " + \
                      f"--train_dataset={root_path}/datasets/cvalues/source/qwen.mindrecord "

        ret = os.system(self.setup_cmd(scripts_cmd, 8))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"

    @pytest.mark.run(order=3)
    def test_qwen2_predict(self):
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/run_dpo.py"

        scripts_cmd = f"{scripts_path} --config={root_path}/model_configs/qwen_config/predict_qwen2_7b.yaml " + \
                      f"--vocab_file={root_path}/checkpoint_download/qwen2/vocab.json " + \
                      f"--merges_file={root_path}/checkpoint_download/qwen2/merges.txt " + \
                      f"--predict_data='hello word' "

        ret = os.system(self.setup_cmd(scripts_cmd, 4))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
