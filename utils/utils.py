import mindspore.communication.management as D
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore import context
import mindspore.common.dtype as mstype
try:
    from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)
from src.pangu_alpha_config import PanguAlphaConfig

def set_pipeline_parallel_context(
    parallel_mode = 'semi_auto_parallel',
    full_batch = True,
    optimizer_shard = False,
    stage_num = 1,
    enable_alltoall = False,
):
    r"""Set pipeline parallel context."""
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    print("rank_id is {}, device_num is {}".format(rank_id, device_num))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(
        parallel_mode=parallel_mode, gradients_mean=False,
        full_batch=bool(full_batch), loss_repeated_mean=True,
        device_num=device_num, enable_parallel_optimizer=bool(optimizer_shard),
        pipeline_stages=stage_num, enable_alltoall=bool(enable_alltoall))
    set_algo_parameters(elementwise_op_strategy_follow=True)
    _set_multi_subgraphs()
    return rank_id, device_num

def get_model_config(model_config, opt):
    D.init()
    device_num = D.get_group_size()
    rank_id = D.get_rank()
    model_parallel_num = model_config.op_level_model_parallel_num
    stage_device_num = int(device_num / model_config.stage_num)
    stage_id = rank_id // stage_device_num
    data_parallel_num = int(stage_device_num / model_parallel_num)
    per_batch_size = model_config.per_batch_size
    batch_size = per_batch_size * data_parallel_num * model_config.micro_size

    micro_batch_interleaved = model_config.micro_batch_interleaved
    recompute_config = TransformerRecomputeConfig(recompute=True,
                                                    recompute_slice_activation=bool(model_config.recompute_slice_activation))
    parallel_config = TransformerOpParallelConfig(data_parallel=data_parallel_num, model_parallel=model_parallel_num,
                                                    pipeline_stage=model_config.stage_num,
                                                    micro_batch_num=model_config.micro_size,
                                                    optimizer_shard=bool(opt.optimizer_shard),
                                                    vocab_emb_dp=bool(opt.word_emb_dp), recompute=recompute_config)
    config = PanguAlphaConfig(batch_size=batch_size // parallel_config.micro_batch_num // micro_batch_interleaved,
                                num_heads=model_config.num_heads, hidden_size=model_config.embedding_size,
                                seq_length=opt.seq_length, vocab_size=opt.vocab_size,
                                use_moe=bool(opt.use_moe), eod_token=opt.eod_id,
                                post_layernorm_residual=model_config.post_layernorm_residual,
                                num_layers=model_config.num_layers, ffn_hidden_size=model_config.embedding_size * 4,
                                eod_reset=bool(opt.eod_reset), load_ckpt_path=opt.load_ckpt_path,
                                param_init_type=mstype.float32 if model_config.param_init_type == 'fp32' else mstype.float16,
                                enable_offload=bool(opt.opt_offload), parallel_config=parallel_config)
    return config