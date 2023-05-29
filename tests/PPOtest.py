"""PPO test"""
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
import mindspore.ops.functional as F
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import ReduceOp
from mindtransformer.models.gpt import GPTModel
# from mindtransformer.models.t5 import T5Model, TransformerNetworkWithLoss

from utils.models.ppo_models import PPO_model, PPOConfig
from mindspore.common.tensor import Tensor
import numpy as np

class Net_test(nn.Cell):
    """BERT trainer"""
    def __init__(self):
        super(Net_test, self).__init__()
        model_config = PPOConfig()
        # model_config.seq_length=9
        self.my_net = PPO_model(model_config)
        
        ckpt_path="/home/jenkins/wangshengnan/0301/transformer/mindformers/models/gpt2/mind_ckpt/gpt2.ckpt"
        ckpt = load_checkpoint(ckpt_path)
        load_param_into_net(self.my_net, ckpt)
        for param in self.my_net.trainable_params():
            print("mp params:", param)
        #     if param.name == "my_net.model.backbone.transformer.encoder.blocks.11.output.mapping.bias":
        #         print(param.name,param.shape, param.value())

    def construct(self, query_tensors, response_tensors, logprobs, values, rewards, attention_mask):
        return self.my_net(query_tensors, response_tensors, logprobs, values, rewards, attention_mask)

bsz = 1
seq_len = 550
max_decode_length = 50

query_tensors = Tensor(np.ones((bsz, seq_len)), mstype.int32)
response_tensors = Tensor(np.ones((bsz, max_decode_length))*2, mstype.int32)
old_logprobs = Tensor(np.ones((bsz, max_decode_length))*0.1, mstype.float32)
old_values = Tensor(np.ones((bsz, max_decode_length))*0.2, mstype.float32)
old_rewards = Tensor(np.ones((bsz, max_decode_length))*0.3, mstype.float32)
attention_mask =  Tensor(np.ones((1, 550)), mstype.float32)

mindspore.set_context(mode=0)

net = Net_test()
output = net(query_tensors, response_tensors, old_logprobs, old_values, old_rewards, attention_mask)
print(output)
