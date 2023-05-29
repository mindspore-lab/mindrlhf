from dataclasses import dataclass
from mindspore.common import dtype as mstype
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.modules.transformer.transformer import default_transformer_config


@dataclass
class PPOConfig:
    """
    PPO config class which defines the model size
    """
    model_type: str = "gpt2"
    from_pretrain: bool = True # if True, get the pretrained ckpt from mindformers
    epochs: int = 50 #50
    total_steps: int = 100000
    batch_size: int = 4 #4
    checkpoint_interval = 10000
    eval_interval: int = 200

    optimizer: str = 'adamw'
    lr: float = 5.0e-6
    betas = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.01

    sceduler_name: str = 'cosine_annealing'
    T_max: int = 100000
    eta_min: float = 5.0e-6

    num_rollouts: int = 128 #128
    chunk_size: int = 16 # 16
    ppo_epochs: int = 4 # n_updates_per_batch
    init_kl_coef: float = 0.1
    target: float = 6.0
    horizon: int = 10000
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.2
    scale_reward: bool = None
    ref_mean: bool = False
    ref_std: bool = False
    gen_experience_kwargs: bool = False

    seq_length: int = 550
    max_prompt_length: int = 500
    max_decode_length: int = 50
    vocab_size: int = 50257
    hidden_size: int = 768
    embedding_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expand_ratio: int = 4
    post_layernorm_residual: bool = False
    dropout_rate: float = 0.1
    compute_dtype: mstype = mstype.float32
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float32
    parallel_config: TransformerOpParallelConfig = default_transformer_config
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    repetition_penalty = 1
    top_k: int = 1
    top_p = 0.95
    is_encoder_decoder = False
    do_sample = False

    # there are some new config parameter introduced in mindformer 0.3.0
    dropout_prob: float = 0.0
    initializer_range: float = 0.02
    eos_token: int = 50256
    expand_ratio: int = 4
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    dtype: mstype = mstype.float32
    param_init_type: str = mstype.float32
    hidden_act: str = 'gelu'
    # checkpoint_name_or_path: str = "/autotest/qianjiahong/rlhf_master/ms/ppo_ckpt/ppo_model_0322_master.ckpt"
    checkpoint_name_or_path: str = "/autotest/qianjiahong/rlhf_master/ms/checkpoint_download/gpt2/gpt2.ckpt"
    column_names = ["query_tensors", "sample_tensors", "logprobs", "values", "rewards"]

    # create experience dataset
    train_dataset_dir = ["../TLDR_data/train/tldr_train_prompts.mindrecord"]
    val_dataset_dir = ["../TLDR_data/val/tldr_val_prompts.mindrecord"]
    columns_to_project = ["prompt_ids", "prompt_mask", "original_sample_ids", "original_sample_mask"]

class RMConfig:
    '''
    Reward Model config class
    '''
    seq_length: int = 550
    compute_dtype: mstype = mstype.float32
    vocab_size: int = 50257
    hidden_size: int = 768
    embedding_size: int = 768 # in mindformers 0.3.0 gpt2 model use parameter name 'embedding_size'
    parallel_config: TransformerOpParallelConfig = default_transformer_config
    batch_size: int = 10
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    layernorm_dtype: mstype = mstype.float32
    softmax_dtype: mstype = mstype.float32
    checkpoint_name_or_path: str = "/autotest/qianjiahong/rlhf_master/ms/rm_ckpt/rw_model_0415.ckpt"
    # there are some new config parameter introduced in mindformer 0.3.0
    dropout_prob: float = 0.0
    initializer_range: float = 0.02
    eos_token: int = 50256
    expand_ratio: int = 4
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    dtype: mstype = mstype.float32
    param_init_type: str = mstype.float32
    hidden_act: str = 'gelu'

@dataclass
class opt:
    ckpt_name_prefix = 'pangu'
    data_column_name = 'input_ids'
    data_url = '/data'
    decay_steps = 200000
    device_num = 8,
    device_target = 'Ascend'
    distribute = True
    enable_alltoall = 0
    end_lr = 1e-06
    eod_id = 8
    pad_id = 6
    eod_reset = 1
    epoch_size = 1
    eval_data_url = None
    eval_steps = 10
    expert_num = 1
    expert_parallel_num = 1
    export = 0
    full_batch = True
    gradient_aggregation_group = 4
    has_trained_epoches = 0
    has_trained_steps = 0
    hccl_connect_time = 6000
    incremental_training = 0
    keep_checkpoint_max = 1
    load_ckpt_name = 'PANGUALPHA3.ckpt'
    load_ckpt_path = None
    offline = 1
    opt_offload = 0
    optimizer = 'adam'
    optimizer_shard = 1
    parallel_mode = 'semi_auto_parallel'
    per_token_num_experts_chosen = 1
    pre_trained = None
    run_type = 'train'
    save_checkpoint = True
    save_checkpoint_path = './'
    save_checkpoint_steps = 2000
    seq_length = 550
    sink_size = 2
    start_lr = 5e-05
    strategy_load_ckpt_path = ''
    tokenizer_path = './tokenizer_path'
    train_and_eval_mode = 0
    train_url = None
    use_moe = 0
    vocab_size = 50432
    warmup_step = 2000
    word_emb_dp = 0

@dataclass
class sft_config:
    mode = '13B'
    num_heads = 40
    num_layers = 20
    embedding_size = 5120
    micro_batch_interleaved = 1
    micro_size = 8
    op_level_model_parallel_num = 4
    per_batch_size = 8
    stage_num = 2
    recompute_slice_activation = 0
    param_init_type = 'fp16',
    post_layernorm_residual = True

@dataclass
class rm_config:
    mode = '2.6B'
    num_heads = 40
    num_layers = 20
    embedding_size = 5120
    micro_batch_interleaved = 1
    micro_size = 8
    op_level_model_parallel_num = 4
    per_batch_size = 8
    stage_num = 2
    recompute_slice_activation = 0
    param_init_type = 'fp16',
    post_layernorm_residual = True
