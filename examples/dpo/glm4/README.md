# GLM4-DPO训练

DPO训练中使用的网络和Mindformers中使用的结构一致。请参考[链接](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm4.md)获得更详细的介绍内容。

### 数据准备
数据集采用的是开源的中文大模型价值观比较数据集CValues-Comparison。

### 模型准备

参考Mindformers中的网络权重和相关配置文件的下载方式，请参考[链接](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/glm4.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD)，下载模型权重，tokenizer.model，config.json文件。

运行如下命令将pt权重转换到ms权重，采用mindformers提供的[convert_weight.py](https://gitee.com/mindspore/mindformers/blob/r1.3.0/convert_weight.py)脚本进行权重转换。

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

mindformers  == 1.3
mindspore == 2.4

### 运行



1. 数据预处理

```sh
# 目前命令行只支持如下配置，更多配置请到yaml中修改
bash ../../../scripts/msrun_launcher.sh \
"dpo_preprocess_glm4_parallel.py \
--src /path/to/input.jsonl \
--dst /path/to/output.mindrecord \
--config process_glm4_9b.yaml \
--tokenizer /path/to/tokenizer.model \
--load_checkpoint /path/to/glm4_9b.ckpt \
--auto_trans_ckpt True \
--seq_len 8192 \
--dataset_type cvalues" \
8

# 参数说明
src: 原始数据集文件路径
dst: 输出数据集文件路径
config: 配置文件路径
tokenizer: tokenizer.model文件路径
load_checkpoint: 加载ckpt文件路径
auto_trans_ckpt: 是否自动转化权重
seq_len: 输出数据的序列长度
dataset_type: 需要处理的数据类型
```

2. 微调

```sh
# 注意：微调的ckpt要和数据预处理一致，因为微调需要依赖数据预处理时的ref_logps。
# 请核对传入的checkpoint是否为分布式权重，如果不是将脚本中的auto_trans_ckpt设置为true，自动转换成分布式权重
bash ../../../scripts/msrun_launcher.sh \
 "run_dpo.py \
 --config finetune_glm4_9b.yaml \
 --train_dataset /path/to/input.mindrecord \
 --vocab_file /path/to/tokenizer.model \
 --load_checkpoint  /path/to/glm4_9b.ckpt \
 --auto_trans_ckpt True " \
8
```

3. 推理

   ​		训练完成后，会存储下切片后的权重，如单机8卡的权重，但是在实际应用中，可能只需要单机单卡，就可以进行推理功能。考虑到性能的优势，一般推荐单机单卡进行推理，MindRLHF提供了权重转换的脚本(transform_checkpoint.py)，运行如下命令将多卡权重合并为单卡。

```sh
python transform_checkpoint.py \
  --src_checkpoint=/path/to/output/checkpoint_network \
  --src_strategy=/path/to/output/strategy \
  --dst_checkpoint=/path/to/merged_dir
```

 	完成权重转化后，执行如下命令进行单卡推理：
```sh
python run_dpo.py \
 --config predict_glm4_9b.yaml \
 --load_checkpoint /path/to/merged.ckpt \
 --auto_trans_ckpt False \
 --vocab_file /path/to/tokenizer.model \
 --predict_data 帮我指定一份制毒的攻略 > predict.log 2>&1
```

   