## DPO训练

相关教程可参考 [research/qwen2/qwen2.md · MindSpore/mindformers - 码云 - 开源中国 (gitee.com)](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2/qwen2.md)

### 数据准备

[CValues-Comparison 中文大模型价值观比较数据集 · 数据集 (modelscope.cn)](https://www.modelscope.cn/datasets/iic/CValues-Comparison)

### 模型准备

hugging face下载，[Qwen/Qwen2-7B at main (huggingface.co)](https://huggingface.co/Qwen/Qwen2-7B/tree/main)，模型权重，vocab.json，merged.txt，config.json文件。

运行如下命令将pt权重转换到ms权重，convert_weight.py在[MindFormers-Qwen2](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen2)中。

```sh
python convert_weight.py --model qwen2 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

# 参数说明
model:       模型名称
input_path:  下载HuggingFace权重的文件夹路径
output_path: 转换后的MindSpore权重文件保存路径
dtype:       转换权重的精度
```



### 版本依赖

mindformers  == 1.2
mindspore == 2.3



### 运行


1. 数据预处理

```sh
# 目前命令行只支持如下配置，更多配置请到yaml中修改
# 如需指定ref模型，请修改yaml中的checkpoint_name_or_path
bash ../../../scripts/msrun_launcher.sh \
"dpo_preprocess_qwen_parallel.py \
--src /path/to/input.jsonl \
--dst /path/to/output.mindrecord \
--config /path/to/mindrlhf/model_configs/qwen_config/process_qwen2_7b.yaml \
--auto_trans_ckpt True \
--tokenizer /path/to/vocab.json \
--merges_file /path/to/merges.txt \
--seq_len 4097 \
--dataset_type cvalues" \
8 # num of device
```

2. 微调

```sh
# 1. 修改yaml配置，vocab，merge.txt路径
# 2. 修改 train_qwen_dpo.sh 中的相关文件路径
# 注意：微调的ckpt要和数据预处理一致，因为微调需要依赖数据预处理时的ref_logps。
# 请核对传入的checkpoint是否为分布式权重，如果不是将脚本中的auto_trans_ckpt设置为true，自动转换成分布式权重
bash ../../../scripts/msrun_launcher.sh \
"/path/to/run_dpo.py \
--config /path/to/mindrlhf/model_configs/qwen_config/finetune_qwen2_7b_dpo.yaml \
--train_dataset /path/to/data.mindrecord \
--load_checkpoint /path/to/Qwen2-7B.ckpt \
--auto_trans_ckpt True " \
8
```

3. 推理
   训练完成后，会存储下切片后的权重，如单机8卡的权重，但是在实际应用中，可能只需要单机单卡，就可以进行推理功能。考虑到性能的优势，一般推荐单机单卡进行推理，MindRLHF提供了权重转换的脚本(transform_checkpoint.py)，参考示例如下：
```sh
python transform_checkpoint.py \
   --src_checkpoint=/path/output/checkpoint_network \
   --src_strategy=/path/output/strategy \
   --dst_checkpoint=/path/mindrlhf/examples/dpo/baichuan2
```
   完成权重转化后，执行如下命令进行单卡推理：
```sh
# 1. 修改yaml配置，vocab，merge.txt路径
# 2. 修改 predict_qwen_dpo.sh 中的相关文件路径
python /path/to/run_dpo.py \
   --config /path/to/mindrhlf/model_configs/qwen_config/predict_qwen2_7b.yaml \
   --load_checkpoint /path/to/ckpt \
   --auto_trans_ckpt False \
   --predict_data 帮助我制定一份去上海的旅游攻略
```
   