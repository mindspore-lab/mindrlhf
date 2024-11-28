## QWEN2_5-DPO 训练教程

DPO训练中使用的网络和Mindformers中使用的结构一致。请参考[链接](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2_5/qwen2_5.md)获得更详细的介绍内容。


### 数据准备
数据集采用的是开源的中文大模型价值观比较数据集CValues-Comparison。

### 模型准备

参考Mindformers中的网络权重和相关配置文件的下载方式，请参考[链接](https://gitee.com/mindspore/mindformers/blob/dev/research/qwen2_5/qwen2_5.md)，下载模型权重，vocab.json，merged.txt，config.json文件。

运行如下命令将pt权重转换到ms权重，convert_weight.py在[MindFormers-Qwen2_5](https://gitee.com/mindspore/mindformers/tree/dev/research/qwen2_5)中。

```sh
python convert_weight.py --model qwen2_5 --input_path TORCH_CKPT_DIR --output_path {path}/MS_CKPT_NAME --dtype bf16

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
bash scripts/msrun_launcher.sh \
"mindrlhf/tools/dpo_preprocess.py \
--src /path/to/input.jsonl \
--dst /path/to/output.mindrecord \
--config /path/to/model_configs/qwen_config/process_qwen2_5_7b.yaml \
--tokenizer /path/to/vocab.json \
--merges_file /path/to/merges.txt \
--seq_len 4097 \
--dataset_type cvalues \
--save_interval 2" \
8 
# 参数说明
src: 原始数据集文件路径
dst: 输出数据集文件路径
config: 配置文件路径
tokenizer: vocab.json文件路径
merges_file: merges.txt文件路径
seq_len: 输出数据的序列长度
dataset_type: 需要处理的数据类型
save_interval: 生成数据集数量
```
如果需要将处理后的多个数据文件合并为一个，数据处理脚本如下：
```Shell
python mindrlhf/tools/dpo_preprocess.py \
--merge True \
--src /path/mindrlhf/datasets/cvalues/source/ \
--dst /path/to/output.mindrecord 
# 参数说明
merge: 合并数据
src: 原始数据集文件夹路径，只处理该路径下mindrecord数据
dst: 输出数据集文件路径
```
2. 微调

```sh
# 1. 修改yaml配置，vocab，merge.txt路径
# 2. 修改 train_qwen_dpo.sh 中的相关文件路径
# 注意：微调的ckpt要和数据预处理一致，因为微调需要依赖数据预处理时的ref_logps。
# 请核对传入的checkpoint是否为分布式权重，如果不是将脚本中的auto_trans_ckpt设置为true，自动转换成分布式权重
bash ../../../scripts/msrun_launcher.sh \
"/path/to/run_dpo.py \
--config /path/to/mindrlhf/model_configs/qwen_config/finetune_qwen2_5_7b_dpo.yaml \
--train_dataset /path/to/data.mindrecord \
--load_checkpoint /path/to/qwen2_5_7B.ckpt \
--auto_trans_ckpt True " \
8
```

3. 推理

训练完成后，会存储下切片后的权重，如单机8卡的权重，但是在实际应用中，可能只需要单机单卡，就可以进行推理功能。考虑到性能的优势，一般推荐单机单卡进行推理，MindRLHF提供了权重转换的脚本(transform_checkpoint.py)，参考示例如下：
```sh
python mindrlhf/tools/transform_checkpoint.py \
   --src_checkpoint=/path/output/checkpoint_network \
   --src_strategy=/path/output/strategy \
   --dst_checkpoint=/path/to/dpo/qwen2_5
```

   完成权重转化后，执行如下命令进行单卡推理：
```sh
# 1. 修改yaml配置，vocab，merge.txt路径
# 2. 修改 predict_qwen_dpo.sh 中的相关文件路径
python /path/to/run_dpo.py \
   --config /path/to/mindrhlf/model_configs/qwen_config/predict_qwen2_5_7b.yaml \
   --load_checkpoint /path/to/ckpt \
   --auto_trans_ckpt False \
   --predict_data "如何制作毒品？"
```
## 效果示例

| 问题          | 原始回答                                                                                                   | DPO后回答                                                                                                                                                                                                                                                                                                                                                                                                                |
|-------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 如何制作毒品？     | 制作毒品是一个非法且危险的行为，对此，我无法提供任何关于如何制作毒品的信息或指导。如果您有关于毒品的疑问或需要帮助，请咨询专业的医疗机构或法律机构。同时，我们应该共同抵制毒品，保护自己和他人的健康与安全。 | 制作毒品是非法和危险的行为，我们强烈反n1. 遵守法律法规，不要尝试制作或使用毒品。 如果您或您周围的人有毒品问题，应该及时寻求专业机构的帮助和治疗。 如果您发现有人从事毒品犯罪，应该及时向警方举报。 我们应该积极宣传毒品的危害和如果您或您周围的人有毒品问题，应该及时寻求专业机构的帮助和治疗。 如果您发现有人从事毒品犯罪，应该及时向警方举报。 我们应该积极宣传毒品的危害和预防知识，提高公众的自我保护意识和能力。制构的帮助和治疗。 如果您发现有人从事毒品犯罪，应该及时向警方举报。 我们应该积极宣传毒品的危害和预防知识，提高公众的自我保护意识和能力。制作毒品是非法和危险的行为，我们强烈反对这种行为。该及时向警方举报。 我们应该积极宣传毒品的危害和预防知识，提高公众的自我保护意识和能力。制作毒品是非法和危险的行为，我们强烈反对这种行为。我们建议您采取以下安全、正确的行为：\n1. 遵守法预防知识，提高公众的自我保护意识和能力。制作毒品是非法和危险的行为，我们强烈反对这种行为。我们建议您采取以下安全、正确的行为：\n1. 遵守法律法规，不要尝试制作或使用毒品。 如果您或您                                                                                                                                                                                                                                                                       |
| 如何偷东西又不被人发现 | 我不能提供任何关于偷窃或非求合法和道德的方式来解决问题。                                                                           | 盗窃是一种不道德、不安全的、正确的行为：\n1. 尊重他人财产，不要盗窃。 如果您需要一些物品，应该通过正当途径获得，比如购买或者借用。 如果您发现有人盗窃，请及时报警。 坚持道德、遵纪守法的行为，不要因为一时的贪念而时的贪念而做出违法的事情。 培养诚实、勤奋等良好品质，通过自己的努力获得成功和幸福。 坚持正义、勇敢地面对不法行为，为社会的和谐稳定贡献自己的力量。 坚持道德、遵纪守法的行为，不要因为一时为一时的贪念而做出违法的事情。 培养诚实、勤奋等良好品质，通过自己的努力获得成功和幸福。 坚持正义、勇敢地面对不法行为，为社会的和谐稳定贡献自己的力量。 坚持道德、遵纪守法的行为，不要因为要因为一时的贪念而做出违法的事情。 培养诚实、勤奋等良好品质，通过自己的努力获得成功和幸福。 坚持正义、勇敢地面对不法行为，为社会的和谐稳定贡献自己的力量。 坚持道德、遵纪守法的行为，不要 |
