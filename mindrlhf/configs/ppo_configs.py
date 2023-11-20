
from dataclasses import dataclass
import mindspore.common.dtype as mstype

@dataclass
class PPOConfig:
    """
    PPO config class which defines the model size
    """
    epochs: int = 2
    total_steps: int = 100000
    batch_size: int = 1
    checkpoint_interval = 10000
    eval_interval: int = 200

    optimizer: str = 'adamw'
    lr: float = 9.0e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    weight_decay: float = 0.01

    sceduler_name: str = 'cosine_annealing'
    T_max: int = 100000
    eta_min: float = 5.0e-6

    num_rollouts: int = 1
    chunk_size: int = 1
    ppo_epochs: int = 1
    init_kl_coef: float = 0.1
    kl_coef: float = 0.02
    target: float = 6.0
    horizon: int = 10000
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 1.0
    pretrain_coef: float = 0.9
    scale_reward: bool = None
    ref_mean: bool = False
    ref_std: bool = False
    gen_experience_kwargs: bool = False

    sink_size: int = 1
    device_target: str = 'Ascend'
    parallel_mode: str = 'semi_auto_parallel'
    full_batch: bool = True
    enable_alltoall: bool = False
    micro_batch_interleaved: int = 1
    start_lr: float = 9e-6
    end_lr: float = 1e-10
    warmup_step: int = 3200
    decay_steps: int = 200000
    opt_offload: bool =False
    mind_dataset_dir: str = "/path/train.mindrecord"
    use_past: bool = False
    inference_micro_size: int = 1
    save_ckpt_dir: str = ""
    save_data_file: str = ""
    sft_model_path: str = "/path/model.yaml"
    critic_model_path: str = "/path/model.yaml"
    reward_model_path: str = "/path/model.yaml"
