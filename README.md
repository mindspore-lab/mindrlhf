<div align="center">

# MindRLHF

[![license](https://img.shields.io/github/license/mindspore-lab/mindrlhf.svg)](https://github.com/mindspore-lab/mindrlhf/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindrlhf)](https://github.com/mindspore-lab/mindrlhf/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindrlhf/pulls)
[![Code style: autopep8](https://img.shields.io/badge/code_style-autopep8-blue)](https://github.com/hhatto/autopep8)

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

* Stage 1: Supervised fine-tuning.
* Stage 2: Reward model training.
* Stage 3: Reinforcement learning training.

MindRLHF integrates the rich model library of the [MindFormers](https://github.com/mindspore-lab/mindformers), providing fine-tuning processes for basic models such as Pangu-Alpha (2.6B, 13B) and GPT-2.

Fully inheriting the parallel interface of MindSpore, MindRLHF can easily deploy models to the training cluster with just one click, enabling training and inference of large models.

To improve inference performance, MindRLHF integrates `incremental inference`, which is known as `K-V cache` or `state reuse` and can achieve more than a 30% improvement in inference performance compared to full inference.

MindRLHF architecture diagram is as follows:

![framework](https://github.com/mindspore-lab/mindrlhf/blob/master/images/framework.jpg)

## Installation
Current version `0.3.0` can be used directly.

There are some requirements for MindRLHF:

|  requirements   | version |
|  ----   |---------|
| MindSpore    | r2.3.1    |
| Mindformers | r1.2.0    |

## Supported Models

Current version of MindRLHF: `0.3.0`

The current version integrates Pangu-alpha(13B), GPT2, Baichuan2(7B/13B) models, and users can explore these two models. In the future, we will provide more models such as LLAMA, BLOOM, GLM, etc. To help users quickly implement their own applications. The specific supported list is shown below:

Table 1： The models and scales supported in MindRLHF
|  Models   | Pangu-alpha |  GPT2   |  Baichuan2   |
|  ----     | ----        |  ----   |  ----        |
| Scales    | 2.6B/13B    | 124M    | 7B/13B       |
| Parallel  | Y           | Y       | Y            |
| Device    | NPU         | NPU     | NPU          |

The support of models for different training stages is shown in the following table:

Table 2： The models and stages supported in MindRLHF
|  Stages     | Pangu-alpha    |  GPT2   |  Baichuan2   |
|  ----       | ----           |  ----   |  ----        |
| SFT         | Y              | Y       | Y            |
| RM          | Y              | Y       | Y            |
| RLHF        | Y              | Y       | Y            |

In the future, we will integrate more models such as LLAMA, GLM, BLOOM, etc.

Now we support `DPO`, and models supported are shown in the following table:

Table 3： The models for DPO
|  Type     |  Baichuan2   |
|  ----       |  ----        |
| offline     | Y            |
| online      |             |

In the future, we will integrate more models such as LLAMA, GLM, Qwen, etc.

## Get Started

* Reward model training: a `GPT2` based reward model training tutorial is listed in 'examples'.

* RLHF fine-tuning: here is an example for RLHF fine-tuning in `MindRLHF`:

```python
ppo_config, sft_model_config, ref_model_config, critic_model_config, rm_model_config = init_configs(
    args)
trainer = PPOTrainer(ppo_config=ppo_config, sft_model_config=sft_model_config, ref_model_config=ref_model_config,
                        critic_model_config=critic_model_config, rm_model_config=rm_model_config)
ppo_with_grad = init_network_and_optimizer(trainer)
rank_id = D.get_rank()
for epoch in range(ppo_config.epochs):
    # sampling
    trainer.make_experience(num_rollouts=ppo_config.num_rollouts)
    dataset = init_ppo_dataset(trainer)
    # use data sink to accelerate
    trainer.train(ppo_with_grad, dataset, epoch)
    trainer.save_checkpoint(rank_id, epoch)
```

## Contribution

Welcome to the community. You can refer to the MindSpore contribution requirements on the Contributor Wiki.

## License

Apache 2.0 License.
