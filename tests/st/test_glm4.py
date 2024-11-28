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
from mindrlhf.models.glm4.glm_dpo import Glm4DPO
from mindrlhf.models.glm4.glm4_tokenizer import ChatGLM4Tokenizer
from mindformers.tools.download_tools import download_with_progress_bar

root_path = os.path.dirname(os.path.abspath(__file__)).split('tests')[0]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
class TestGlm4DPO:
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
    def test_glm4_dpo_process(self):
        download_with_progress_bar("https://www.modelscope.cn/models/ZhipuAI/glm-4-9b/resolve/master/tokenizer.model",
                                   f"{root_path}/checkpoint_download/glm4/tokenizer.model")

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/mindrlhf/tools/dpo_preprocess.py"

        scripts_cmd = f"{scripts_path} --src={root_path}/datasets/cvalues/source/one.jsonl " + \
                      f"--dst={root_path}/datasets/cvalues/source/glm.mindrecord " + \
                      f"--config={root_path}/model_configs/glm_config/process_glm4_9b.yaml " + \
                      f"--tokenizer={root_path}/checkpoint_download/glm4/tokenizer.model " + \
                      f"--seq_len=8192 " + \
                      f"--dataset_type=cvalues " + \
                      f"--save_interval=2"
        ret = os.system(self.setup_cmd(scripts_cmd, 8))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
        os.system(f"python {root_path}/mindrlhf/tools/dpo_preprocess.py \
                    --merge True --src={root_path}/datasets/cvalues/source/ \
                    --dst {root_path}/datasets/cvalues/source/glm.mindrecord")

        assert os.path.isfile(f"{root_path}/datasets/cvalues/source/glm.mindrecord")

    @pytest.mark.run(order=2)
    def test_glm4_finetune(self):
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/run_dpo.py"

        scripts_cmd = f"{scripts_path} --config={root_path}/model_configs/glm_config/finetune_glm4_9b.yaml " + \
                      f"--train_dataset={root_path}/datasets/cvalues/source/glm.mindrecord "

        ret = os.system(self.setup_cmd(scripts_cmd, 8))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"

    @pytest.mark.run(order=3)
    def test_glm4_predict(self):
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = f"{root_path}/run_dpo.py"

        scripts_cmd = f"{scripts_path} --config={root_path}/model_configs/glm_config/predict_glm4_9b.yaml " + \
                      f"--vocab_file={root_path}/checkpoint_download/glm4/tokenizer.model " + \
                      f"--predict_data='hello word' "
        ret = os.system(self.setup_cmd(scripts_cmd, 1))
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log/worker_*.log"
