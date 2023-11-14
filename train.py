import os
import time
import argparse
import mindspore
from mindspore import context
import mindspore.communication.management as D
from mindrlhf.trainer.ppo_trainer import PPOTrainer
from mindrlhf.utils.configs import init_configs, init_network_and_optimizer, init_dataset
from mindrlhf.utils.utils import set_pipeline_parallel_context
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
    parser.add_argument(
        '--dataset_dir',
        default='/path/dataset.mindrecord',
        help='dataset_dir (str): dataset dir.')
    parser.add_argument(
        '--sft_model_path',
        default='/path/sft_model.yaml',
        help='sft_model_path (str): sft model yaml path.')
    parser.add_argument(
        '--critic_model_path',
        default='/path/critic_model.yaml',
        help='critic_model_path (str): critic model yaml path.')
    parser.add_argument(
        '--reward_model_path',
        default='/path/reward_model.yaml',
        help='reward_model_path (str): reward model yaml path.')
    args_opt = parser.parse_args()
    return args_opt

def run_rlhf(args):
    context.set_context(save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path, mode=args.mode, 
                        device_target=args.device_target, enable_compile_cache=False, 
                        compile_cache_path="./cache", max_call_depth=4096,
                        memory_optimize_level='O1', max_device_memory=args.max_device_memory)

    ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config = init_configs(args)

    set_pipeline_parallel_context(parallel_mode=ppo_config.parallel_mode, full_batch=ppo_config.full_batch, 
                                  stage_num=sft_model_config.parallel_config.pipeline_stage,
                                  enable_alltoall=ppo_config.enable_alltoall)

    trainer = PPOTrainer(ppo_config=ppo_config, 
                         sft_model_config=sft_model_config,
                         ref_model_config=ref_model_config,
                         critic_model_config=critic_model_config,
                         rm_model_config=rm_model_config)
    ppo_with_loss, optimizer, update_cell = init_network_and_optimizer(trainer)
    ppo_with_grad = TrainOneStepWithLossScaleCell(ppo_with_loss, optimizer=optimizer, config=sft_model_config,
                                                  scale_update_cell=update_cell, enable_global_norm=True)
    rank_id = D.get_rank()
    for epoch in range(ppo_config.epochs):
        # Sampling
        trainer.make_experience(num_rollouts=ppo_config.num_rollouts)
        dataset = init_dataset(trainer)
        # Use data sink to accelerate
        trainer.ppo_model.policy_model.set_train()
        trainer.ppo_model.critic_model.set_train()
        sink_process = mindspore.data_sink(ppo_with_grad, dataset, sink_size=ppo_config.sink_size)
        steps = dataset.dataset_size // ppo_config.sink_size
        for batch in range(steps):
            for i in range(ppo_config.ppo_epochs):
                out = sink_process()
                print("PPO Batch: {} | PPO Epoch: {} | loss: {} | lr: {} | is overflow: {} | loss scale: {}"
                      .format(batch, i, out[0], out[1], out[2], out[3]), flush=True)
        # Save checkpoints
        trainer.save_checkpoint(rank_id, epoch)
    print("PPO train done!")

if __name__ == "__main__":
    args = get_args()
    run_rlhf(args)
