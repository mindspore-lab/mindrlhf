export GLOG_v=3
bash /path/run_distribute_train.sh \
/path/temp.mindrecord \
/path/hccl_8p.json \
0 \
8 \
/path/model_configs/baichuan_config/run_baichuan2_7b.yaml \
/path/model_configs/baichuan_config/run_baichuan2_7b.yaml
