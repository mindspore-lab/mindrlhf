import 
import time
import math
import copy
import numpy as np
from mindspore import context, Tensor, Parameter
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype

from mindspore.nn.wrap.cell_wrapper import PipelineCell, _VirtualDatasetCell, MicroBatchInterleaved
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.ops import operations as P
try:
    from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)

import numpy as np
from dataclasses import dataclass
import mindspore.common.dtype as mstype
import mindspore
from mindspore import nn
from mindspore import ops

from ppo_trainer import AcceleratePPOTrainer
from ppo_models import PPOConfig

from src.adam import AdamWeightDecayOp
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell
from src.utils import LearningRate, FP32StateAdamWeightDecay
from utils import set_pipeline_parallel_context
from dataset import IteratorStore
from mindspore.dataset import GeneratorDataset

from mindformers.tools.register import MindFormerConfig
from mindformers.core.parallel_config import build_parallel_config

from mindformers import AutoConfig


@dataclass
class opt:
    device_target = 'Ascend'
    parallel_mode = 'semi_auto_parallel'
    full_batch = True
    enable_alltoall = False
    micro_batch_interleaved = 1
    start_lr = 1e-04
    end_lr = 1e-05
    warmup_step = 500
    decay_steps = 200000
    opt_offload =False
    optimizer = 'adam'
    mind_dataset_dir = " "
    use_past = False
    inference_micro_size = 1

def set_weight_decay(params):
    """
    Set weight decay coefficient, zero for bias and layernorm, 1e-1 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
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

context.set_context(save_graphs=False, save_graphs_path='./graph', mode=context.GRAPH_MODE, 
                    device_target=opt.device_target, enable_compile_cache=False, 
                    compile_cache_path="./cache", max_call_depth=4096,
                    memory_optimize_level='O1')


sft_model_path = "xxx.yaml"
critic_cfg_path = "xxx.yaml"
reward_cfg_path = "xxx.yaml"

config = MindFormerConfig(sft_model_path)
build_parallel_config(config)
sft_model_config = AutoConfig.from_pretrained(sft_model_path)
sft_model_config.parallel_config = copy.deepcopy(config.parallel_config)
sft_model_config.use_past = opt.use_past
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


ppo_config = PPOConfig()


if opt.use_past:
    sft_model_config.batch_size = ppo_config.chunk_size
    ref_model_config.batch_size = ppo_config.chunk_size
    critic_model_config.batch_size = ppo_config.chunk_size
    rm_model_config.batch_size = ppo_config.chunk_size

print("[ACT Configure] is: ", sft_model_config, sft_model_config.parallel_config, flush=True)
print("[REF Configure] is: ", ref_model_config, ref_model_config.parallel_config, flush=True)
print("[CRT Configure] is: ", critic_model_config, critic_model_config.parallel_config, flush=True)
print("[RM Configure] is: ", rm_model_config, rm_model_config.parallel_config, flush=True)

set_pipeline_parallel_context(parallel_mode=opt.parallel_mode, full_batch=opt.full_batch,
    optimizer_shard=sft_model_config.parallel_config.optimizer_shard, 
    stage_num=sft_model_config.parallel_config.pipeline_stage, enable_alltoall=opt.enable_alltoall)
print("parallel model: ", opt.parallel_mode)

ppo_config.seq_length = sft_model_config.seq_length

trainer = AcceleratePPOTrainer(ppo_config=ppo_config, 
                               sft_model_config=sft_model_config,
                               ref_model_config=ref_model_config,
                               critic_model_config=critic_model_config,
                               rm_model_config=rm_model_config,
                               opt=opt)
train_dataset = trainer.prompt_dataloader
test_dataset = trainer.prompt_dataloader

epochs = PPOConfig.epochs
batch_size = PPOConfig.batch_size
learning_rate = PPOConfig.lr


'''unfrozen_layers = 4
for param in trainer.ppo_model.trainable_params():
    param_name = param.name
    for i in range(sft_model_config.num_layers - unfrozen_layers):
        if param_name.startswith("blocks.{}".format(i)):
            param.requires_grad = False'''
        
'''for param in trainer.ppo_model.trainable_params():
    param_name = param.name
    if "blocks.{}.layernorm1.gamma".format(unfrozen_layers) in param_name:
        break
    else:
        param.requires_grad = False'''

ppo_with_loss_net = trainer.ppo_model
ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)
lr = LearningRate(learning_rate=opt.start_lr, end_learning_rate=opt.end_lr,
                  warmup_steps=opt.warmup_step, decay_steps=opt.decay_steps)
params = ppo_with_loss.trainable_params()
group_params = set_weight_decay(params)

if opt.optimizer == "lamb":
    optimizer = nn.Lamb(group_params, learning_rate=lr)
elif opt.opt_offload:
    optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95,
                                  param_init_type=sft_model_config.param_init_type)
else:
    optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)


loss_scale_value = math.pow(2, 12)
update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)

ppo_with_grad = PanguAlphaTrainOneStepWithLossScaleCell(
    ppo_with_loss, optimizer=optimizer, config=sft_model_config, scale_update_cell=update_cell)

def train_loop(ppo_model, dataset, ppo_with_grad):
    trainer.ppo_model.policy_model.set_train()
    trainer.ppo_model.critic_model.set_train()
    iterator = dataset.create_dict_iterator()
    for batch, databatch in enumerate(iterator):
        for i in range(ppo_model.ppo_config.ppo_epochs):
            print(f"=====PPO Epoch {i}=====")
            query_tensors = Tensor(databatch["query_tensors"], mstype.int32)
            sample_tensors = Tensor(databatch["response_tensors"], mstype.int32)
            old_logprobs = Tensor(databatch["logprobs"], mstype.float32)
            old_values = Tensor(databatch["values"], mstype.float32)
            old_rewards = Tensor(databatch["rewards"], mstype.float32)
            advantages = Tensor(databatch["advantages"], mstype.float32)
            returns = Tensor(databatch["returns"], mstype.float32)
            pretrain_ids = Tensor(databatch["pretrain_ids"], mstype.int32)
            loss_mask = Tensor(databatch["loss_mask"], mstype.float32)
            response = databatch["response_tensors"].asnumpy()
            attention_mask = np.not_equal(response, PPOConfig.pad_token_id)
            max_prompt_length = ppo_model.ppo_config.max_prompt_length
            for i in range(attention_mask.shape[0]):
                last_index = np.max(np.where(attention_mask[i] > 0))
                attention_mask[i, max_prompt_length-1] = 1
                attention_mask[i, last_index] = 0.0
            attention_mask[:, :max_prompt_length]
            attention_mask = Tensor(attention_mask, mstype.float32)

            step_begin_time = time.time()
            print("input shapes: ", query_tensors.shape, sample_tensors.shape, old_logprobs.shape, old_values.shape, old_rewards.shape,
                                    attention_mask.shape, advantages.shape, returns.shape, pretrain_ids.shape, loss_mask.shape, flush=True)
            loss, cond, _ = ppo_with_grad(query_tensors, sample_tensors, old_logprobs, old_values, old_rewards,
                                          attention_mask, advantages, returns, pretrain_ids, loss_mask)
            print("Step elapsed time: ", time.time()-step_begin_time, flush=True)
            print("is overflow: ", cond, flush=True)

            loss = loss.asnumpy()
            print("loss: ", loss)
            ppo_model.post_backward_callback()

epochs = PPOConfig.epochs
for t in range(epochs):
    ep_begin_time = time.time()
    print("Epoch {}, begin at {} \n------------------------------- ".format(t+1, ep_begin_time), flush=True)
    trainer.make_experience(num_rollouts=PPOConfig.num_rollouts)
    end_time = time.time()
    print("Make_experience, end at {}, elapsed time {} \n------------------------------- ".format(end_time, end_time-ep_begin_time), flush=True)

    pipeline = IteratorStore(trainer.store)
    dataset = GeneratorDataset(pipeline, column_names=["query_tensors", "response_tensors", "logprobs", "values"
                                                       "rewards", "advantages", "returns", "pretrain_ids", "loss_mask"])
    print("ppo update batch_size: ", PPOConfig.batch_size, flush=True)

    # TODO: fix batch_size when dp != 1
    # dataset = dataset.batch(batch_size=sft_model_config.batch_size * sft_model_config.parallel_config.micro_batch_num)
    dataset = dataset.batch(batch_size=PPOConfig.batch_size \
        * sft_model_config.parallel_config.data_parallel \
        * sft_model_config.parallel_config.micro_batch_num)
    tl_begin_time = time.time()
    print("Train_loop, begin at {} \n------------------------------- ".format(tl_begin_time), flush=True)
    train_loop(trainer.ppo_model, dataset, ppo_with_grad)
    end_time = time.time()
    print("Epoch {}, end at {}, elapsed time {} \n------------------------------- ".format(t+1, end_time, end_time-tl_begin_time), flush=True)

    rank_id = D.get_rank()
    mindspore.save_checkpoint(trainer.ppo_model.policy_model, 
                              "./checkpoint_ppo/policy_model_device_{}_epoch_{}.ckpt".format(rank_id, t), 
                              integrated_save=False)

print("Done!")

