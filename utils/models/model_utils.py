from .ppo_models import PPO_model, CausalLMHydraWithValueHead
from .reward_model import RewardModel


def init_models(config, rm_config):
    ppo_model = PPO_model(config)
    ref_model = CausalLMHydraWithValueHead(config)
    reward_model = RewardModel(rm_config)
    return ppo_model, ref_model, reward_model

