execute_path=$(pwd)
bash /path/run_distribute_reward.sh \
     "python3 reward_train.py \
     --config /path/model_configs/baichuan_config/run_baichuan2_7b_rm.yaml \
     --train_dataset /path/data.mindrecord " \
     /path/hccl_8p.json [0,8] 8 