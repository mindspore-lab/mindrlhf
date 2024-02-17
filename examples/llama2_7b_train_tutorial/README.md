# Llama2_7b 训练教程

## 训练脚本执行

### 脚本参数介绍
训练llama2_7b，框架提供了当前目录下的`run_distribute_train.sh`脚本，用户可以通过执行该脚本来完成训练。该脚本的参数如下：
- DATA_DIR：数据集路径。 如果第一次训练可以参考`examples/rlhf_train_tutorial/README.md`下的描述来生成数据集。
- RANK_TABLE_FILE：rank_table_file请参考mindformers中的生成方法。
- RANK_START：从第几个rank开始训练。这里rank一般指第几张GPU/NPU卡。
- LOCAL_DEVICE_NUM：集群中GPU/NPU卡的总数。
- SFT_MODEL_PATH：SFT模型的yaml路径。如`model_configs/llama2_config/llama2_7b.yaml`的绝对路径。
- REWARD_MODEL_PATH：reward模型和critic模型的yaml路径。如`model_configs/llama2_config/llama2_7b.yaml`的绝对路径。

### 脚本执行方法
以8卡为例，执行以下命令即可开始训练：
```Shell
bash examples/llama2_7b_train_tutorial/run_distribute_train.sh /path/to/dataset.mindrecord /path/to/hccl_8p_01234567_127.0.0.1.json 0 8 /path/to/llama2_7b.yaml /path/to/llama2_7b.yaml
```