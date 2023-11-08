import copy
from mindformers.tools.register import MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers import AutoConfig

__all__ = ['combine_config', 'init_configs']

def combine_config(ppo_config, model_config):
    new_config = copy.deepcopy(ppo_config)
    new_config.seq_length = model_config.seq_length
    new_config.max_decode_length = model_config.max_decode_length
    new_config.max_prompt_length = new_config.seq_length - new_config.max_decode_length
    return new_config

def init_configs(ppo_config):
    # ppo_config = PPOConfig()
    sft_model_path = ppo_config.sft_model_path
    critic_cfg_path = ppo_config.critic_cfg_path
    reward_cfg_path = ppo_config.reward_cfg_path

    config = MindFormerConfig(sft_model_path)
    build_parallel_config(config)
    sft_model_config = AutoConfig.from_pretrained(sft_model_path)
    sft_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    sft_model_config.use_past = ppo_config.use_past
    ref_model_config = AutoConfig.from_pretrained(sft_model_path)
    ref_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    ref_model_config.use_past = False

    config = MindFormerConfig(critic_cfg_path)
    build_parallel_config(config)
    critic_model_config = AutoConfig.from_pretrained(critic_cfg_path)
    critic_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    critic_model_config.use_past = False

    config = MindFormerConfig(reward_cfg_path)
    build_parallel_config(config)
    rm_model_config = AutoConfig.from_pretrained(reward_cfg_path)
    rm_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    rm_model_config.use_past = False

    if ppo_config.use_past:
        sft_model_config.batch_size = ppo_config.chunk_size
        ref_model_config.batch_size = ppo_config.chunk_size
        critic_model_config.batch_size = ppo_config.chunk_size
        rm_model_config.batch_size = ppo_config.chunk_size

    ppo_config = combine_config(ppo_config, sft_model_config)

    print("[ACT Configure] is: ", sft_model_config, sft_model_config.parallel_config, flush=True)
    print("[REF Configure] is: ", ref_model_config, ref_model_config.parallel_config, flush=True)
    print("[CRT Configure] is: ", critic_model_config, critic_model_config.parallel_config, flush=True)
    print("[RM Configure] is: ", rm_model_config, rm_model_config.parallel_config, flush=True)

    return ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config