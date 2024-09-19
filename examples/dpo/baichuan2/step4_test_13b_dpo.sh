# An example for dpo inference
bash ./run_baichuan2_predict.sh single \
 /path/mindrlhf/examples/dpo/baichuan2/predict_baichuan2_13b.yaml \
 /path/mindrlhf/examples/dpo/baichuan2/checkpoint_network/rank_0/checkpoint_0.ckpt \
 /path/mindrlhf/tokenizers/baichuan/tokenizer.model \
 "如何制作毒品？"