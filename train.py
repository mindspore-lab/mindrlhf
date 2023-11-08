import os
import time
import math
import argparse
from mindspore import context
import mindspore.communication.management as D
import mindspore.nn as nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
import mindspore
from mindspore import nn
from mindspore.dataset import GeneratorDataset

from mindrlhf.trainer.ppo_trainer import PPOTrainer, set_weight_decay
from mindrlhf.configs.ppo_configs import PPOConfig
from mindrlhf.utils.configs import init_configs
from mindrlhf.utils.utils import set_pipeline_parallel_context
from mindrlhf.utils.adam import AdamWeightDecayOp
from mindrlhf.utils.utils import LearningRate, FP32StateAdamWeightDecay
from mindrlhf.utils.dataset import IteratorStore
from mindrlhf.wrapper import TrainOneStepWithLossScaleCell


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--align_type',
        default="rlhf",
        help='the name for align algorithm. Currently, It supports rlhf and dpo')
    parser.add_argument(
        '--model',
        default="pangu",
        help='model name or path for align model. Currently, It supports pangu, gpt, bloom, llama')
    parser.add_argument(
        '--device_target',
        default='Ascend',
        help='device_target (str): Ascend.')
    parser.add_argument(
        '--mode',
        default=0,
        help='run mode (int): Running in GRAPH_MODE(0) or PYNATIVE_MODE(1).')
    parser.add_argument(
        '--save_graphs',
        default=False,
        help='save_graphs (bool): True or False.')
    parser.add_argument(
        '--save_graphs_path',
        default='./graph',
        help='save_graphs_path (str): the path to save graphs.')
    parser.add_argument(
        '--enable_compile_cache',
        default=False,
        help='enable_compile_cache (bool): Whether to save or load the cache of the graph compiled by front-end')
    parser.add_argument(
        '--max_device_memory',
        default='47GB',
        help='max_device_memory (str): Set the maximum memory available for devices. The format is xxGB.')
    args_opt = parser.parse_args()
    return args_opt

def run_rlhf(args):
    context.set_context(save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path, mode=args.mode, 
                        device_target=args.device_target, enable_compile_cache=False, 
                        compile_cache_path="./cache", max_call_depth=4096,
                        memory_optimize_level='O1', max_device_memory=args.max_device_memory)

    ppo_config = PPOConfig()
    ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config = init_configs(ppo_config)

    set_pipeline_parallel_context(parallel_mode=ppo_config.parallel_mode, full_batch=ppo_config.full_batch, 
        stage_num=sft_model_config.parallel_config.pipeline_stage, enable_alltoall=ppo_config.enable_alltoall)
    print("parallel model: ", ppo_config.parallel_mode)

    ppo_config.seq_length = sft_model_config.seq_length

    trainer = PPOTrainer(ppo_config=ppo_config, 
                                    sft_model_config=sft_model_config,
                                    ref_model_config=ref_model_config,
                                    critic_model_config=critic_model_config,
                                    rm_model_config=rm_model_config)

    ppo_with_loss_net = trainer.ppo_model
    ppo_with_loss = _VirtualDatasetCell(ppo_with_loss_net)
    lr = LearningRate(learning_rate=ppo_config.start_lr, end_learning_rate=ppo_config.end_lr,
                    warmup_steps=ppo_config.warmup_step, decay_steps=ppo_config.decay_steps)
    params = ppo_with_loss.trainable_params()
    # print("trainable_params", trainer.ppo_model.trainable_params())
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

    ppo_with_grad = TrainOneStepWithLossScaleCell(
        ppo_with_loss, optimizer=optimizer, config=sft_model_config, scale_update_cell=update_cell, enable_global_norm=True)

    for t in range(ppo_config.epochs):
        # Sampling
        ep_begin_time = time.time()
        print("Epoch {}, begin at {} \n------------------------------- "
            .format(t+1, time.strftime('%H:%M:%S', time.localtime(ep_begin_time))), flush=True)
        trainer.make_experience(num_rollouts=ppo_config.num_rollouts)
        end_time = time.time()
        print("Make experience, end at {}, elapsed time {} \n------------------------------- "
            .format(time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-ep_begin_time), flush=True)

        pipeline = IteratorStore(trainer.store)
        dataset = GeneratorDataset(pipeline, column_names=["query_tensors", "response_tensors", "logprobs",
                                                        "values", "rewards", "advantages", "returns",
                                                        "pretrain_ids", "loss_mask", "attention_mask"])
        print("ppo update batch_size: ", ppo_config.batch_size, flush=True)

        # Training
        dataset = dataset.batch(batch_size=ppo_config.batch_size \
            * sft_model_config.parallel_config.data_parallel \
            * sft_model_config.parallel_config.micro_batch_num)

        rank_id = D.get_rank()
        device_num = D.get_group_size()
        # use data sink to accelerate
        trainer.ppo_model.policy_model.set_train()
        trainer.ppo_model.critic_model.set_train()
        sink_process = mindspore.data_sink(ppo_with_grad, dataset, sink_size=ppo_config.sink_size)
        steps = dataset.dataset_size // ppo_config.sink_size
        tl_begin_time = time.time()
        print("Train loop, begin at {} \n------------------------------- "
            .format(time.strftime('%H:%M:%S', time.localtime(tl_begin_time))), flush=True)
        for batch in range(steps):
            for i in range(ppo_config.ppo_epochs):
                out = sink_process()
                print("PPO Batch: {} | PPO Epoch: {} | loss: {} | lr: {} | is overflow: {} | loss scale: {}"
                    .format(batch, i, out[0], out[1], out[2], out[3]), flush=True)
        end_time = time.time()
        print("Epoch {}, end at {}, elapsed time {} \n------------------------------- "
            .format(t+1, time.strftime('%H:%M:%S', time.localtime(end_time)), end_time-tl_begin_time), flush=True)

        # save_dir = ""
        save_dir = ppo_config.save_dir
        if save_dir:
            print("Save checkpoints in {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            mindspore.save_checkpoint(trainer.ppo_model.policy_model, 
                                    os.path.join(save_dir + "/rank_{}".format(rank_id), "policy_model_epoch_{}.ckpt".format(t)), 
                                    integrated_save=False)

    print("PPO train done!")

if __name__ == "__main__":
    args = get_args()
    run_rlhf(args)
