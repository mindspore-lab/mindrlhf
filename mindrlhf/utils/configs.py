import copy
import math
from mindspore import nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.dataset import GeneratorDataset
from mindformers.tools.register import MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers import AutoConfig
from mindrlhf.utils.dataset import IteratorStore
from mindrlhf.trainer.ppo_trainer import set_weight_decay
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.configs.ppo_configs import PPOConfig

__all__ = ['combine_config', 'init_configs']

def combine_config(ppo_config, model_config):
    new_config = copy.deepcopy(ppo_config)
    new_config.seq_length = model_config.seq_length
    new_config.max_decode_length = model_config.max_decode_length
    new_config.max_prompt_length = new_config.seq_length - new_config.max_decode_length
    return new_config

def init_configs(args=None):
    '''init ppo_configs'''
    ppo_config = PPOConfig()
    if args:
        ppo_config.mind_dataset_dir = args.dataset_dir
        ppo_config.sft_model_path = args.sft_model_path
        ppo_config.critic_model_path = args.critic_model_path
        ppo_config.reward_model_path = args.reward_model_path
    sft_model_path = ppo_config.sft_model_path
    critic_model_path = ppo_config.critic_model_path
    reward_model_path = ppo_config.reward_model_path

    config = MindFormerConfig(sft_model_path)
    build_parallel_config(config)
    sft_model_config = AutoConfig.from_pretrained(sft_model_path)
    sft_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    sft_model_config.use_past = ppo_config.use_past
    ref_model_config = AutoConfig.from_pretrained(sft_model_path)
    ref_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    ref_model_config.use_past = False

    config = MindFormerConfig(critic_model_path)
    build_parallel_config(config)
    critic_model_config = AutoConfig.from_pretrained(critic_model_path)
    critic_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    critic_model_config.use_past = False

    config = MindFormerConfig(reward_model_path)
    build_parallel_config(config)
    rm_model_config = AutoConfig.from_pretrained(reward_model_path)
    rm_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    rm_model_config.use_past = False

    if ppo_config.use_past:
        sft_model_config.batch_size = ppo_config.chunk_size
        ref_model_config.batch_size = ppo_config.chunk_size
        critic_model_config.batch_size = ppo_config.chunk_size
        rm_model_config.batch_size = ppo_config.chunk_size

    ppo_config = combine_config(ppo_config, sft_model_config)
    ppo_config.seq_length = sft_model_config.seq_length
    print("[ACT Configure] is: ", sft_model_config, sft_model_config.parallel_config, flush=True)
    print("[REF Configure] is: ", ref_model_config, ref_model_config.parallel_config, flush=True)
    print("[CRT Configure] is: ", critic_model_config, critic_model_config.parallel_config, flush=True)
    print("[RM Configure] is: ", rm_model_config, rm_model_config.parallel_config, flush=True)

    return ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config

def init_network_and_optimizer(trainer):
    '''init network and optimizer'''
    ppo_with_loss_net = trainer.ppo_model
    ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)
    ppo_config = trainer.ppo_config
    lr = LearningRate(learning_rate=ppo_config.start_lr, end_learning_rate=ppo_config.end_lr,
                      warmup_steps=ppo_config.warmup_step, decay_steps=ppo_config.decay_steps)
    params = ppo_with_loss.trainable_params()
    group_params = set_weight_decay(params)

    if ppo_config.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    elif ppo_config.opt_offload:
        optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=ppo_config.eps, beta1=ppo_config.beta1,
                                      beta2=ppo_config.beta2, param_init_type=sft_model_config.param_init_type)
    else:
        optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=ppo_config.beta1, 
                                             beta2=ppo_config.beta2, eps=ppo_config.eps)

    loss_scale_value = math.pow(2, 12)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)
    return ppo_with_loss, optimizer, update_cell

def init_dataset(trainer):
    pipeline = IteratorStore(trainer.store)
    dataset = GeneratorDataset(pipeline, column_names=["query_tensors", "response_tensors", "logprobs",
                                                        "values", "rewards", "advantages", "returns",
                                                        "pretrain_ids", "loss_mask", "attention_mask"])
    ppo_config = trainer.ppo_config
    sft_model_config = trainer.sft_model_config
    print("ppo update batch_size: ", ppo_config.batch_size, flush=True)
    # Training
    dataset = dataset.batch(batch_size=ppo_config.batch_size * sft_model_config.parallel_config.data_parallel * \
                            sft_model_config.parallel_config.micro_batch_num)
    return dataset