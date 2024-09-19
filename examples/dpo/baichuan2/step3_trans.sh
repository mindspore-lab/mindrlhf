# trans the ckpt to a unified one
python transform_checkpoint.py \
  --src_checkpoint=/path/output/checkpoint_network \
  --src_strategy=/path/output/strategy \
  --dst_checkpoint=/path/mindrlhf/examples/dpo/baichuan2
