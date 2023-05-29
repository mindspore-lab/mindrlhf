import datetime
import json
import glob
import os
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
from mindspore.train.serialization import load_distributed_checkpoint, load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
try:
    from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig

except ImportError as e:
    print("Import ERROR, expect mindformers to be installed. "
          "Please refer to the page https://gitee.com/mindspore/mindformers.git to install the mindformers.")
    print("Now exit the program.")
    exit(1)

import numpy as np
import mindspore.common.dtype as mstype
import mindspore
from mindspore import nn
from mindspore import ops
# Download data from open datasets
#from download import download
from ppo_trainer_pangu import PanguPPOTrainer
from mindspore import context

from src.adam import AdamWeightDecayOp
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.utils import LearningRate, FP32StateAdamWeightDecay
from utils.utils import set_pipeline_parallel_context, get_model_config
from utils.dataset import IteratorDataset
from utils.configs import opt, sft_config, rm_config, PPOConfig
from mindspore.dataset import GeneratorDataset

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)



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

context.set_context(save_graphs=2, save_graphs_path='./graph', mode=context.GRAPH_MODE, 
                    device_target=opt.device_target, enable_compile_cache=True, 
                    compile_cache_path="./cache")

sft_model_config = get_model_config(sft_config, opt)
rm_model_config = get_model_config(rm_config, opt)

print("[SFT Configure] is: ", sft_model_config, flush=True)
print("[RM Configure] is: ", rm_model_config, flush=True)

PPOConfig.batch_size = sft_model_config.batch_size

set_pipeline_parallel_context(parallel_mode=opt.parallel_mode, full_batch=opt.full_batch,
    optimizer_shard=opt.optimizer_shard, stage_num=sft_model_config.parallel_config.pipeline_stage, enable_alltoall=opt.enable_alltoall)

trainer = PanguPPOTrainer(ppo_config=PPOConfig, 
                               sft_model_config=sft_model_config,
                               rm_model_config=rm_model_config,
                               opt=opt)
train_dataset = trainer.prompt_dataloader
test_dataset = trainer.prompt_dataloader


def IsLastStage(opt):
    device_num = D.get_group_size()
    rank = D.get_rank()
    stage_num = opt.stage_num
    per_stage_num = int(device_num / stage_num)
    is_last_stage = device_num - 1 - per_stage_num < rank  and rank <= device_num - 1
    print("IsLastStage: ", is_last_stage, flush=True)
    return is_last_stage


epochs = PPOConfig.epochs
batch_size = PPOConfig.batch_size
learning_rate = PPOConfig.lr
trainer.generate_experience(num_rollouts=PPOConfig.num_rollouts)


if sft_config.stage_num > 1:
    print("pipeline cell")
    ppo_with_loss_net = PipelineCell(MicroBatchInterleaved(trainer.ppo_model,
                                                           sft_config.micro_batch_interleaved),
                                    sft_model_config.parallel_config.micro_batch_num)
else:
    print("non-pipeline cell")
    ppo_with_loss_net = trainer.ppo_model

ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)

lr = LearningRate(learning_rate=opt.start_lr, end_learning_rate=opt.end_lr,
                  warmup_steps=opt.warmup_step, decay_steps=opt.decay_steps)
params = ppo_with_loss.trainable_params()
group_params = set_weight_decay(params)

print("trainable_params: ", group_params)
set_pipeline_parallel_context(parallel_mode=opt.parallel_mode, full_batch=opt.full_batch,
    optimizer_shard=opt.optimizer_shard, stage_num=sft_model_config.parallel_config.pipeline_stage, enable_alltoall=opt.enable_alltoall)
print("parallel model: ", opt.parallel_mode)

if opt.optimizer == "lamb":
    optimizer = nn.Lamb(group_params, learning_rate=lr)
elif opt.opt_offload:
    optimizer = AdamWeightDecayOp(group_params, learning_rate=lr, eps=1e-8, beta1=0.9, beta2=0.95,
                                  param_init_type=sft_model_config.param_init_type)
else:
    optimizer = FP32StateAdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)


loss_scale_value = math.pow(2, 32)
update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value, scale_factor=2, scale_window=1000)


if sft_config.stage_num > 1:
    print("pipeline cell")
    ppo_with_grad = PanguAlphaTrainPipelineWithLossScaleCell(
        ppo_with_loss, optimizer=optimizer, config=sft_model_config, scale_update_cell=update_cell)
else:
    print("non-pipeline cell")
    ppo_with_grad = PanguAlphaTrainOneStepWithLossScaleCell(
        ppo_with_loss, optimizer=optimizer, config=sft_model_config, scale_update_cell=update_cell)

def train_loop(ppo_model, dataset, ppo_with_grad):
    ppo_with_grad.set_train()
    iterator = dataset.create_dict_iterator()
    for batch, databatch in enumerate(iterator):
        for i in range(ppo_model.ppo_config.ppo_epochs):
            query_tensors = Tensor(databatch["query_tensors"], mstype.int32)
            sample_tensors = Tensor(databatch["response_tensors"], mstype.int32)
            old_logprobs = Tensor(databatch["logprobs"], mstype.float32)
            old_values = Tensor(databatch["values"], mstype.float32)
            old_rewards = Tensor(databatch["rewards"], mstype.float32)

            attention_mask = ops.not_equal(sample_tensors, PPOConfig.pad_token_id)
            attention_mask = ops.Cast()(attention_mask, mstype.float32)
    
            print("shapes: ", query_tensors.shape, sample_tensors.shape, old_logprobs.shape, old_values.shape, old_rewards.shape, flush=True)
            set_pipeline_parallel_context(parallel_mode=opt.parallel_mode, full_batch=opt.full_batch,
                optimizer_shard=opt.optimizer_shard, stage_num=sft_model_config.parallel_config.pipeline_stage, enable_alltoall=opt.enable_alltoall)

            loss, _, _ = ppo_with_grad(query_tensors, sample_tensors, old_logprobs, old_values, old_rewards, attention_mask)

            loss = loss.asnumpy()
            print("loss: ", loss)
        if IsLastStage(sft_config):
            context.set_auto_parallel_context(full_batch=True, pipeline_stages=1)
            ppo_model.post_backward_callback()

    # TODO: post_epoch_callback()


epochs = PPOConfig.epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    print("trainer.ppo_elements", len(trainer.ppo_elements))
    pipeline = IteratorDataset(trainer.ppo_elements)
    dataset = GeneratorDataset(pipeline, column_names=["query_tensors", "response_tensors", "logprobs",
                                                       "values", "rewards"])
    print("batch_size: ", PPOConfig.batch_size)
    dataset = dataset.batch(batch_size=PPOConfig.batch_size)
    train_loop(trainer.ppo_model, dataset, ppo_with_grad)
    trainer.generate_experience(num_rollouts=PPOConfig.num_rollouts)
    # test_loop(model, test_dataset, loss_fn)
print("Done!")

