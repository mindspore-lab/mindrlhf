
import mindspore.nn as nn
from mindformers.models.bloom import BloomLMHeadModel, BloomConfig
from mindformers import LlamaForCausalLM, LlamaConfig
from mindformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
from mindformers.models.pangualpha import PanguAlphaHeadModel, PanguAlphaConfig
from mindrlhf.models.baichuan2.baichuan2_7b import Baichuan7BV2ForCausalLM


class BaseModel(nn.Cell):
    '''BaseModel'''
    _model_list = ['pangu', 'bloom', 'baichuan2_7b', 'baichuan2_13b', 'gpt2', 'llama']

    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def select_actor_model(self, model_config):
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in actor/reference model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError("only support {}".format(' '.join(self._model_list)))
        if self.model_type == 'pangu':
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
            self.lm_head = self.model.head
        elif self.model_type == 'bloom':
            self.model = BloomLMHeadModel(model_config)
            self.backbone = self.model.transformer
            self.lm_head = self.model.head
        elif self.model_type == 'baichuan2_7b':
            self.model = Baichuan7BV2ForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head
        elif self.model_type == 'baichuan2_13b':
            self.model = Baichuan13BV2ForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head
        elif self.model_type == 'gpt2':
            self.model = GPT2LMHeadModel(model_config)
            self.backbone = self.model.backbone
            self.lm_head = self.model.head
        elif self.model_type == 'llama':
            self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
            self.lm_head = self.model.lm_head

    def select_critic_model(self, model_config):
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in critic model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError("only support {}".format(' '.join(self._model_list)))
        if self.model_type == 'pangu':
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == 'bloom':
            self.model = BloomLMHeadModel(model_config)
            self.backbone = self.model.transformer
        elif self.model_type == 'baichuan2_7b':
            self.model = Baichuan7BV2ForCausalLM(model_config)
            self.backbone = self.model.model
        elif self.model_type == 'baichuan2_7b':
            self.model = Baichuan13BV2ForCausalLM(model_config)
            self.backbone = self.model.model
        elif self.model_type == 'gpt2':
            self.model = GPT2LMHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == 'llama':
            self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model

    def select_reward_model(self, model_config):
        self.model_type = None
        if not model_config.model_name:
            raise NotImplementedError("model_name in reward model is None")
        for model in self._model_list:
            if model in model_config.model_name:
                self.model_type = model
        if not self.model_type:
            raise NotImplementedError("only support {}".format(' '.join(self._model_list)))
        if self.model_type == 'pangu':
            self.model = PanguAlphaHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == 'bloom':
            self.model = BloomLMHeadModel(model_config)
            self.backbone = self.model.transformer
        elif self.model_type == 'baichuan2_7b':
            self.model = Baichuan7BV2ForCausalLM(model_config)
            self.backbone = self.model.model
        elif self.model_type == 'baichuan2_13b':
            self.model = Baichuan13BV2ForCausalLM(model_config)
            self.backbone = self.model.model
        elif self.model_type == 'gpt2':
            self.model = GPT2LMHeadModel(model_config)
            self.backbone = self.model.backbone
        elif self.model_type == 'llama':
            self.model = LlamaForCausalLM(model_config)
            self.backbone = self.model.model
