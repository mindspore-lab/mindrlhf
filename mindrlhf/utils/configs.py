import copy
import math
from dataclasses import asdict, make_dataclass
import mindspore
import mindspore.nn as nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.dataset import GeneratorDataset, MindDataset
from mindspore.dataset.transforms import TypeCast
from mindformers.tools.register import MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config
from mindformers import AutoConfig
from mindrlhf.configs.ppo_configs import PPOConfig
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.utils.dataset import IteratorStore
from mindrlhf.wrapper import TrainOneStepWithLossScaleCell, TrainPipelineWithLossScaleCell

__all__ = ['combine_config', 'init_configs']


def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    def decay_filter(x): return 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-1
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def combine_config(ppo_config, model_config):
    config_temp = asdict(ppo_config)
    for k, v in vars(model_config).items():
        if k not in config_temp:
            config_temp[k] = v
    config_temp['max_prompt_length'] = config_temp['seq_length'] - config_temp['max_decode_length']
    PPOConfig = make_dataclass("PPOConfig", [(key, type(value)) for key, value in config_temp.items()])
    return PPOConfig(**config_temp)


def init_configs(args=None):
    ppo_config = PPOConfig()
    if args:
        ppo_config.mind_dataset_dir = args.dataset_dir
        ppo_config.sft_model_path = args.sft_model_path
        ppo_config.reward_model_path = args.reward_model_path
        ppo_config.critic_model_path = args.critic_model_path
        ppo_config.save_data_file = args.save_data_file
        ppo_config.align_type = args.align_type
    sft_model_path = ppo_config.sft_model_path
    critic_model_path = ppo_config.critic_model_path
    reward_model_path = ppo_config.reward_model_path

    config = MindFormerConfig(sft_model_path)
    build_parallel_config(config)
    sft_model_config = AutoConfig.from_pretrained(sft_model_path)
    sft_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    sft_model_config.parallel = copy.deepcopy(config.parallel)
    sft_model_config.model_name = config.trainer.model_name
    ppo_config.use_past = config.model.model_config.use_past

    ref_model_config = AutoConfig.from_pretrained(sft_model_path)
    ref_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    ref_model_config.parallel = copy.deepcopy(config.parallel)
    ref_model_config.use_past = False
    ref_model_config.model_name = config.trainer.model_name

    config = MindFormerConfig(critic_model_path)
    build_parallel_config(config)
    critic_model_config = AutoConfig.from_pretrained(critic_model_path)
    critic_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    critic_model_config.parallel = copy.deepcopy(config.parallel)
    critic_model_config.use_past = False
    critic_model_config.model_name = config.trainer.model_name

    config = MindFormerConfig(reward_model_path)
    build_parallel_config(config)
    rm_model_config = AutoConfig.from_pretrained(reward_model_path)
    rm_model_config.parallel_config = copy.deepcopy(config.parallel_config)
    rm_model_config.parallel = copy.deepcopy(config.parallel)
    rm_model_config.use_past = False
    rm_model_config.model_name = config.trainer.model_name

    if ppo_config.use_past:
        sft_model_config.batch_size = ppo_config.chunk_size
        ref_model_config.batch_size = ppo_config.chunk_size
        critic_model_config.batch_size = ppo_config.chunk_size
        rm_model_config.batch_size = ppo_config.chunk_size
    ppo_config.model_name = sft_model_config.model_name
    ppo_config = combine_config(ppo_config, sft_model_config)
    print("[PPO Configure] is: ", ppo_config, flush=True)
    print("[ACT Configure] is: ", sft_model_config, sft_model_config.parallel_config, flush=True)
    print("[REF Configure] is: ", ref_model_config, ref_model_config.parallel_config, flush=True)
    print("[CRT Configure] is: ", critic_model_config, critic_model_config.parallel_config, flush=True)
    print("[RM Configure] is: ", rm_model_config, rm_model_config.parallel_config, flush=True)

    return ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config


def init_network_and_optimizer(trainer):
    '''init network and optimizer'''
    sft_model_config = trainer.sft_model_config
    ppo_config = trainer.ppo_config
    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        ppo_with_loss_net = PipelineCell(MicroBatchInterleaved(trainer.ppo_model,
                                                               ppo_config.micro_batch_interleaved),
                                         sft_model_config.parallel_config.micro_batch_num)
    else:
        print("non-pipeline cell")
        ppo_with_loss_net = trainer.ppo_model
    ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)
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
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2, scale_window=1000)

    if sft_model_config.parallel_config.pipeline_stage > 1:
        print("pipeline cell")
        ppo_with_grad = TrainPipelineWithLossScaleCell(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                       scale_update_cell=update_cell)
    else:
        print("non-pipeline cell")
        ppo_with_grad = TrainOneStepWithLossScaleCell(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                      scale_update_cell=update_cell, enable_global_norm=True)
    return ppo_with_grad


def init_ppo_dataset(trainer):
    ppo_config = trainer.ppo_config
    sft_model_config = trainer.sft_model_config
    column_names = ["query_tensors", "response_tensors", "logprobs",
                    "values", "rewards", "advantages", "returns",
                    "pretrain_ids", "loss_mask", "attention_mask"]
    if ppo_config.save_data_file and 'stages' in ppo_config.align_type:
        dataset = MindDataset(dataset_files=ppo_config.save_data_file, shuffle=False)
        dataset = dataset.project(columns=column_names)
    else:
        pipeline = IteratorStore(trainer.store)
        dataset = GeneratorDataset(pipeline, column_names=column_names)
    type_cast_op_int32 = TypeCast(mindspore.int32)
    type_cast_op_fp16 = TypeCast(mindspore.float16)
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="query_tensors")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="response_tensors")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="logprobs")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="values")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="rewards")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="advantages")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="returns")
    dataset = dataset.map(operations=type_cast_op_int32, input_columns="pretrain_ids")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="loss_mask")
    dataset = dataset.map(operations=type_cast_op_fp16, input_columns="attention_mask")
    dataset = dataset.batch(batch_size=ppo_config.batch_size
                            * sft_model_config.parallel_config.data_parallel)
    return dataset
