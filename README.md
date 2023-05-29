<div align="center">

# MindRLHF

[![CI](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/pyversions/mindcv)](https://pypi.org/project/mindcv)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mindcv.readthedocs.io/en/latest)
[![license](https://img.shields.io/github/license/mindspore-lab/mindcv.svg)](https://github.com/mindspore-lab/mindcv/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindcv)](https://github.com/mindspore-lab/mindrlhf/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindrlhf/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

English | [中文](README_CN.md)

[Introduction](#introduction) |
[Installation](#installation) |
[Supported Models](#supported-models) |
[Get Started](#get-started) |
[Contributions](#Contributions) |
[License](#License)

</div>

# Introduction

OPENAI's [ChatGPT](https://openai.com/blog/chatgpt) has demonstrated astonishing natural language processing capabilities, opening the door to universal artificial intelligence. Its exceptional performance is closely tied to the [Reinforcement Learning from Human Feedback](https://openai.com/research/learning-from-human-preferences) (RLHF) algorithm. In its predecessor, [InstructGPT](https://openai.com/research/instruction-following), RLHF was used to collect human feedback and generate content that better aligns with human cognition and values, thus compensating for potential cognitive biases in large models.

MindSpore RLHF (MindRLHF) is based on the [MindSpore](https://gitee.com/mindspore/mindspore) and utilizes the framework's capabilities for large model parallel training, inference, and deployment to help customers quickly train and deploy RLHF algorithm processes with models that have billions or trillions of parameters.

The MindRLHF learning process consists of three stages:

Stage 1: Supervised fine-tuning.
Stage 2: Reward model training.
Stage 3: Reinforcement learning training.

MindRLHF integrates the rich model library of the [MindFormers](https://github.com/mindspore-lab/mindformers), providing fine-tuning processes for basic models such as Pangu-Alpha (2.6B, 13B) and GPT-2.

Fully inheriting the parallel interface of MindSpore, MindRLHF can easily deploy models to the training cluster with just one click, enabling training and inference of large models.

To improve inference performance, MindRLHF integrates `incremental inference`, which is known as `K-V cache` or `state reuse` and can achieve more than a 30% improvement in inference performance compared to full inference.

MindRLHF architecture diagram is as follows:

![架构图](https://github.com/mindspore-lab/mindrlhf/blob/master/images/framework.jpg)

## Installation

## Supported Models

Current version of MindRLHF: `0.1.0`

The current version integrates Pangu-alpha(13B) and GPT2 models, and users can explore these two models. In the future, we will provide more models such as LLAMA, BLOOM, GLM, etc. To help users quickly implement their own applications. The specific supported list is shown below:

Table 1： The models and scales supported in MindRLHF
|  Models   | Pangu-alpha |  GPT2   |
|  ----   | ----        |  ----   |
| Scales    | 2.6B/13B    | 124M    |
| Parallel | Y          | N       |
| Device    | NPU         | NPU     |

The support of models for different training stages is shown in the following table:

Table 2： The models and stages supported in MindRLHF
|  Stages     | Pangu-alpha    |  GPT2   |
|  ----        | ----           |  ----   |
| SFT| Y              | Y       |
| RM  | Y              | N       |
| RLHF  | Y              | Y       |

In the future, we will integrate more models such as LLAMA, GLM, BLOOM, etc.

## Get Started

Here is an example for RLHF fine-tuning with `GPT2` in `MindRLHF`:

```python
import mindspore
from mindspore import context
from ppo_trainer import MindPPOTrainer, train_loop
from utils.models.ppo_models import PPOConfig, PPOTrainOneStepCell
from utils.models.reward_model import RMConfig
from dataset import create_ppo_dataset, create_experience_dataset
from utils.optimizer import init_optimizer
from utils.models.model_utils import init_models


context.set_context(device_target='Ascend', device_id=0, mode=mindspore.GRAPH_MODE)

# Initialize config
config = PPOConfig()
rm_config = RMConfig()
# Initialize dataset
train_dataset, eval_dataset = create_experience_dataset(config)
# Initialize models
ppo_model, ref_model, reward_model = init_models(config, rm_config)
# Initialize optimizer
optimizer = init_optimizer(ppo_model, config)
ppo_with_grad = PPOTrainOneStepCell(ppo_model, optimizer)
# Initialize ppo
trainer = MindPPOTrainer(config, rm_config, train_dataset, eval_dataset, ppo_model, ref_model, reward_model,
                         optimizer, ppo_with_grad)

for t in range(config.epochs):
    print(f"Epoch {t}\n-------------------------------")
    # experience collect
    trainer.generate_experience(num_rollouts=config.num_rollouts)
    # PPO dataset
    dataset = create_ppo_dataset(trainer.ppo_elements, config)
    # PPO train
    trainer.train(dataset)
    if t % 5 == 4:
        # eval
        reward_mean = trainer.evaluate()
        print("reward_mean is ", reward_mean)
print("RLHF Train and Eval Done!")
```

## Contribution

Welcome to the community. You can refer to the MindSpore contribution requirements on the Contributor Wiki.

## License

Apache 2.0 License.
