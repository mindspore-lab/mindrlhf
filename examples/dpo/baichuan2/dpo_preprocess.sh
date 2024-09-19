# preprocess for offline DPO
export GLOG_v=2

bash ./scripts/msrun_launcher.sh \
"dpo_preprocess_baichuan_parallel.py \
--src /path/mindrlhf/datasets/cvalues/source/one.jsonl \
--dst /path/mindrlhf/data/cvalues/temp/cvalues_one_4096.mindrecord \
--config /path/mindrlhf/model_configs/baichuan_config/process_baichuan2_13b.yaml \
--tokenizer /path/mindrlhf/tokenizers/baichuan/tokenizer.model \
--seq_len 4097 \
--dataset_type cvalues" \
8 # num of device
