<div align="center">

# MindRLHF

[![license](https://img.shields.io/github/license/mindspore-lab/mindrlhf.svg)](https://github.com/mindspore-lab/mindrlhf/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindrlhf)](https://github.com/mindspore-lab/mindrlhf/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindrlhf/pulls)
[![Code style: autopep8](https://img.shields.io/badge/code_style-autopep8-blue)](https://github.com/hhatto/autopep8)

[English](README.md) | 中文

[简介](#简介) |
[安装](#安装) |
[支持列表](#支持列表) |
[快速入门](#快速入门) |
[教程](#教程) |
[贡献](#贡献) |
[许可证](#许可证)

</div>

## 简介

OPENAI的[ChatGPT](https://openai.com/blog/chatgpt)在自然语言方面表现出了令人震惊的效果，开启了通用人工智能的序幕,它的优秀表现，与 RLHF（[Reinforcement Learning from Human Feedback](https://openai.com/research/learning-from-human-preferences)）算法密不可分。在ChatGPT的前身[InstructGPT](https://openai.com/research/instruction-following)中，利用RLHF算法，通过收集人类反馈的信息，可以生成更加符合人类认知和价值观的内容，从而弥补了大模型中潜在的认知偏差。

`MindSpore RLHF`（简称 `MindRLHF`）以[MindSpore](https://gitee.com/mindspore/mindspore)作为基础框架，利用框架具备的大模型并行训练、推理、部署等能力，助力客户快速训练及部署带有百亿、千亿级别基础模型的RLHF算法流程。MindRLHF包含3个阶段的学习流程：
* 阶段1： 预训练模型训练
* 阶段2： 奖励模型训练
* 阶段3： 强化学习训练

MindRLHF集成了大模型套件[MindFormers](https://github.com/mindspore-lab/mindformers)中丰富的模型库， 提供了Pangu-Alpha(2.6B, 13B)、GPT-2等基础模型的微调流程。MindRLHF 完全继承MindSpore的并行接口，可以一键将模型部署到训练集群上，开启大模型的训练和推理。

为了提升推理性能， MindRLHF中集成了`增量推理`，通过状态复用，相比于全量推理，推理性能可提升`30%`以上。

MindRLHF架构图如下：

![framework](https://github.com/mindspore-lab/mindrlhf/blob/master/images/framework.jpg)

## 安装
当前版本`0.3.0`无需安装，用户下载即可使用。
当前版本所依赖框架:
|  依赖   | 版本|
|  ----   | ----        |
| MindSpore    | r2.2   |
| Mindformers | r0.8    |


## 支持列表

当前 MindRLHF 版本：`0.3.0`

当前版本集成了Pangu-alpha(13B)、GPT2、Baichuan2(7B/13B) 模型，用户可以基于这两个模型进行探索。未来，我们将提供更多模型如LLAMA、BLOOM、GLM等，帮助用户快速实现自己的应用。具体支持列表如下所示：

表 1： 当前MindSpore RLHF支持的模型和规模
|  模型   | Pangu-alpha |  GPT2   | Baichuan2 |
|  ----   | ----        |  ----   |  ----   |
| 规模    | 2.6B/13B    | 124M    | 7B/13B    |
| 支持并行 | Y          | Y       | Y       |
| 硬件    | NPU         | NPU     | NPU     |

当前流程下，不同模型对不同训练阶段的支持情况如下表所示：

表 2： 当前MindSpore RLHF支持的模型和阶段
|  训练阶段     | Pangu-alpha    |  GPT2   |  Baichuan2   |
|  ----        | ----           |  ----   |  ----   |
| 预训练模型训练| Y              | Y       | Y       |
| 奖励模型训练  | Y              | Y       | Y       |
| 强化学习训练  | Y              | Y       | Y       |

未来，我们将打通更多的模型，如`LLAMA`、`GLM`、`BLOOM`等，敬请期待。

## 快速入门

* 奖励模型训练: 在`examples`文件夹中展示了如何结合`GPT2`进行奖励模型微调的过程。

* RLHF 微调: 下面是`MindRLHF`使用模型进行微调的过程，示例代码如下：

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


## 贡献

欢迎参与社区贡献，可参考MindSpore贡献要求Contributor Wiki。

## 许可证

Apache 2.0许可证
