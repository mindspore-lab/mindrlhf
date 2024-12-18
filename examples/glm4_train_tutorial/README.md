# GLM4-PPO 训练教程

### 数据准备

数据集采用的是开源的中文大模型价值观比较数据集[CValues-Comparison](https://www.modelscope.cn/datasets/damo/CValues-Comparison/summary)。

### 模型准备

参考Mindformers中的网络权重和相关配置文件的下载方式，请参考[链接](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm4.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD)
，下载模型权重，tokenizer.model，config.json文件。

运行如下命令将pt权重转换到ms权重，采用mindformers提供的[convert_weight.py](https://gitee.com/mindspore/mindformers/blob/r1.3.0/convert_weight.py)
脚本进行权重转换。

```sh
python convert_weight.py --model glm-n \
--input_path path/to/TORCH_CKPT_DIR/ \
--output_path path/to/ms.ckpt \
--dtype bf16

# 参数说明
# model:       模型名称
# input_path:  下载HuggingFace权重的文件夹路径，注意最后面有/
# output_path: 转换后的MindSpore权重文件保存路径
# dtype:       转换权重的精度
```

### 推荐依赖版本

mindformers == dev
mindspore == 2.4.1

### 运行

1. 数据预处理

```sh
cd examples/rlhf_train_tutorial
python rlhf_data.py
--tokenizer_name_or_path glm4_9b
--file_path path/to/train.jsonl
--output_path path/to/cvalues_8192.mindrecord
--max_prompt_length 4096
--seq_length 8193
--pad_token_id 151329


# 参数说明
tokenizer_name_or_path：编码使用的 tokenizer 名称或 tokenizer 文件对应路径。目前仅支持基于 mindformers 实现的 tokenizer。
file_path：原始数据文件。当前仅支持 json 格式文件。
output_path：输出 mindrecord 文件路径。
max_prompt_length：prompt_id paading 后的目标长度。
seq_length：pretrain_id 与 loss_mask padding 后的目标长度。
pad_token_id：padding 时使用的 pad_token_id。
```

2. 训练
   用户可通过run_glm4_rlhf.sh脚本，启动ppo训练，运行命令为：

```sh
bash examples/glm4_train_tutorial/run_glm4_rlhf.sh \
path/to/model_configs/glm4_config/predict_glm4_9b_chat.yaml \
path/to/model_configs/glm4_config/predict_glm4_9b_chat.yaml \
path/to/model_configs/glm4_config/predict_glm4_9b_chat.yaml \
path/to/model_configs/glm4_config/finetune_glm4_9b.yaml \
path/to/model_configs/glm4_config/finetune_glm4_9b.yaml \
path/to/model_configs/glm4_config/finetune_glm4_9b.yaml \
path/to/ppo_data/ppo_data.mindrecord \
path/to/cvalues_8192.mindrecord \
False \
./ckpt \
4 \
True \
./ckpt \
path/to/rm_ckpt \
path/to/cm_ckpt \
path/to/ref_ckpt \


# 脚本参数介绍
# G_SFT_PATH：生成数据阶段sft_model模型配置路径
# G_RM_PATH：生成数据阶段reward_model模型配置路径
# G_CRITIC_PATH：生成数据阶段critic_model模型配置路径
# T_SFT_PATH：训练阶段sft_model模型配置路径
# T_RM_PATH：训练阶段reward_model模型配置路径
# T_CRITIC_PATH：训练阶段critic_model模型配置路径
# SAVE_DATA_FILE：保存文件路径
# DATASET_FILE：加载数据文件路径
# ONLY_SAVE_STRATEGY：是否只保存策略文件
# SAVE_CKPT_DIR：保存权重文件路径
# COMPILE_CACHE：是否开启compile_cache
# DEV_NUM：使用卡数
# USE_PARALLEL：是否启动并行
# SFT_CKPT：加载sft_model权重路径
# RM_CKPT：加载reward_model权重路径
# CRITIC_CKPT：加载critic_model权重路径
# REF_CKPT：加载ref_model权重路径
```



   