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
from dataclasses import dataclass
import mindspore.common.dtype as mstype
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset
# Download data from open datasets
#from download import download
from ppo_trainer_pangu import PanguPPOTrainer
from utils.models.ppo_models_pangu import PPOConfig, PPO_model, AdaptiveKLController
from mindspore import context

from src.adam import AdamWeightDecayOp
from src.dataset import create_dataset
from src.pangu_alpha import PanGUAlphaWithLoss, PanguAlphaModel, PanguAlpha_Model, PanGuHead
from src.pangu_alpha_wrapcell import PanguAlphaTrainOneStepWithLossScaleCell, PanguAlphaTrainPipelineWithLossScaleCell
from src.pangu_alpha_config import set_parse, PanguAlphaConfig
from src.utils import LearningRate, get_args, FP32StateAdamWeightDecay
from src.utils import download_data
from src.callbacks import EvalCallBack, LossCallBack
from src.metrics import PPLMetric
from src.tokenization_jieba import JIEBATokenizer
from mindspore.ops.operations._inner_ops import Send, Receive
from utils import set_pipeline_parallel_context, get_model_config
from dataset import IteratorStore
from mindspore.dataset import GeneratorDataset

np.set_printoptions(threshold=np.inf)

project_root = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


'''opt = get_args()
set_parse(opt)
opt.seq_length = PPOConfig.seq_length
print('opt: ', opt)'''

@dataclass
class opt:
    ckpt_name_prefix = 'pangu'
    data_column_name = 'input_ids'
    data_url = '/data'
    decay_steps = 200000
    device_num = 8
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
    padding_id = 6
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
    vocab_size = 40000
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
    op_level_model_parallel_num = 8
    per_batch_size = 8
    stage_num = 1
    recompute_slice_activation = 0
    param_init_type = 'fp16',
    post_layernorm_residual = True

@dataclass
class rm_config:
    mode = '13B'
    num_heads = 40
    num_layers = 40
    embedding_size = 5120
    micro_batch_interleaved = 1
    micro_size = 1
    op_level_model_parallel_num = 8
    per_batch_size = 1
    stage_num = 1
    recompute_slice_activation = 0
    param_init_type = 'fp16',
    post_layernorm_residual = True


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
                    device_target=opt.device_target, enable_compile_cache=True, 
                    compile_cache_path="./cache")

sft_model_config = get_model_config(sft_config, opt)
rm_model_config = get_model_config(rm_config, opt)
rm_model_config.dropout_rate = 0.0

print("[SFT Configure] is: ", sft_model_config, flush=True)
print("[RM Configure] is: ", rm_model_config, flush=True)

PPOConfig.batch_size = sft_model_config.batch_size

set_pipeline_parallel_context(parallel_mode=opt.parallel_mode, full_batch=opt.full_batch,
    optimizer_shard=opt.optimizer_shard, stage_num=rm_model_config.parallel_config.pipeline_stage, enable_alltoall=opt.enable_alltoall)

PPOConfig.chunk_size = 1
trainer = PanguPPOTrainer(ppo_config=PPOConfig, 
                               sft_model_config=sft_model_config,
                               rm_model_config=rm_model_config,
                               opt=opt)

tokenizer = JIEBATokenizer(os.path.join('/autotest/shiwenqi/tokenizer/', 'vocab.model'))

trainer.reward_fn.load("/autotest/shiwenqi/mindspore-chatgpt-ckpt/rm_ckpt/iter_0000700/merged/rm_model.ckpt")


sample_1 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_2 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_3 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_4 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_5 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_6 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_7 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_8 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'
sample_9 = 'User: \n\n \n\nEvaluate:\n\nAssistant: \n\nscore'

samples = [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8, sample_9]
tokens = []
for i in range(len(samples)):
    token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(samples[i]))
    print(len(token))
    tokens.append(token+(550-len(token))*[tokenizer.pad_id])
tokens = Tensor(tokens, mstype.int32)

print("tokens: ", tokens)
scores, original_scores, norms_scores = trainer.reward_fn(tokens, tokens)

context.reset_auto_parallel_context()
all_scores = trainer.sr_net_2(scores)
print("all_scores: ", all_scores, flush=True)