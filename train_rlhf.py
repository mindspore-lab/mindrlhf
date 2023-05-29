#!/usr/bin/env python
# coding: utf-8
import time
import mindspore
from mindspore import context
from ppo_trainer import MindPPOTrainer
from utils.models.ppo_models import PPOTrainOneStepCell
from utils.configs import RMConfig, PPOConfig
from utils.dataset import create_ppo_dataset, create_experience_dataset
from utils.optimizer import init_optimizer
from utils.models.model_utils import init_models


context.set_context(device_target='Ascend', device_id=0, mode=mindspore.GRAPH_MODE)
# context.set_context(save_graphs=True, save_graphs_path='./graph_train_in/')

config = PPOConfig()
rm_config = RMConfig()
train_dataset, eval_dataset = create_experience_dataset(config)
ppo_model, ref_model, reward_model = init_models(config, rm_config)
optimizer = init_optimizer(ppo_model, config)
ppo_with_grad = PPOTrainOneStepCell(ppo_model, optimizer)
trainer = MindPPOTrainer(config, rm_config, train_dataset, eval_dataset, ppo_model, ref_model, reward_model,
                         optimizer, ppo_with_grad)

for t in range(config.epochs):
    print(f"Epoch {t}\n-------------------------------")
    start0 = time.time()
    trainer.generate_experience(num_rollouts=config.num_rollouts)
    end = time.time()
    print("generate_experience time: ", end-start0)
    dataset = create_ppo_dataset(trainer.ppo_elements, config)
    start = time.time()
    trainer.train(dataset)
    end = time.time()
    print("train time: ", end-start)
    if t % 5 == 4:
        start = time.time()
        reward_mean = trainer.evaluate()
        end = time.time()
        print("evaluate time: ", end-start)
        print("reward_mean is ", reward_mean)
    end = time.time()
    print("Per epoch time: ", end-start0)
print("RLHF Train and Eval Done!")
