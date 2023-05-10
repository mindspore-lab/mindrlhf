<div align="center">

# MindRLHF

[![CI](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml/badge.svg)](https://github.com/mindspore-lab/mindcv/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/pyversions/mindcv)](https://pypi.org/project/mindcv)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mindcv.readthedocs.io/en/latest)
[![license](https://img.shields.io/github/license/mindspore-lab/mindcv.svg)](https://github.com/mindspore-lab/mindcv/blob/main/LICENSE.md)
[![open issues](https://img.shields.io/github/issues/mindspore-lab/mindcv)](https://github.com/mindspore-lab/mindrlhf/issues)
[![PRs](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/mindspore-lab/mindrlhf/pulls)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](README.md) | 中文(README.md)

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

## 安装
TODO

## 支持列表

当前 MindRLHF 版本：`0.1.0`

当前版本集成了Pangu-alpha(13B)、GPT2模型，用户可以基于这两个模型进行探索。未来，我们将提供更多模型如LLAMA、BLOOM、GLM等，帮助用户快速实现自己的应用。具体支持列表如下所示：

表 1： 当前MindSpore RLHF支持模型和规模
|  模型   | Pangu-alpha |  GPT2   |
|  ----   | ----        |  ----   |
| 规模    | 2.6B/13B    | 124M    |
| 支持并行 | Y          | N       |
| 硬件    | NPU         | NPU     |

当前流程下，不同模型对不同训练阶段的支持情况如下表所示：

表 2： 当前MindSpore RLHF支持模型和规模
|  训练阶段     | Pangu-alpha    |  GPT2   |
|  ----        | ----           |  ----   |
| 预训练模型训练| Y              | Y       |
| 奖励模型训练  | Y              | N       |
| 强化学习训练  | Y              | Y       |

未来，我们将打通更多的模型，如`LLAMA`、`GLM`、`BLOOM`等，敬请期待。

## 快速入门

下面是`MindRLHF`使用`GPT2`进行微调的过程，示例代码如下：

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

# 初始化配置
config = PPOConfig()
rm_config = RMConfig()
# 初始化数据集
train_dataset, eval_dataset = create_experience_dataset(config)
# 初始化模型
ppo_model, ref_model, reward_model = init_models(config, rm_config)
# 初始化优化器
optimizer = init_optimizer(ppo_model, config)
ppo_with_grad = PPOTrainOneStepCell(ppo_model, optimizer)
# 初始化PPO训练
trainer = MindPPOTrainer(config, rm_config, train_dataset, eval_dataset, ppo_model, ref_model, reward_model,
                         optimizer, ppo_with_grad)

for t in range(config.epochs):
    print(f"Epoch {t}\n-------------------------------")
    # 经验收集
    trainer.generate_experience(num_rollouts=config.num_rollouts)
    # PPO 数据重建
    dataset = create_ppo_dataset(trainer.ppo_elements, config)
    # PPO 训练
    trainer.train(dataset)
    if t % 5 == 4:
        # 评估
        reward_mean = trainer.evaluate()
        print("reward_mean is ", reward_mean)
print("RLHF Train and Eval Done!")
```


## 贡献

欢迎参与社区贡献，可参考MindSpore贡献要求Contributor Wiki。

## 许可证

Apache 2.0许可证
