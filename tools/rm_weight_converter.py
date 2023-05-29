import torch
import numpy as np
from mindspore import save_checkpoint, Tensor

path = "/autotest/shiwenqi/mindspore-chatgpt-ckpt/rm_ckpt/iter_0000700/"
# dir_list = ["mp_rank_00", "mp_rank_01", "mp_rank_02", "mp_rank_03", "mp_rank_04", "mp_rank_05", "mp_rank_06", "mp_rank_07"]
dir_list = ["merged"]
for d in dir_list:
    torch_path = path + d + "/rm_model.pt"
    mindspore_path = path + d + "/rm_model.ckpt"
    state_dict = torch.load(torch_path, map_location='cpu')

    param_list = []

    for key in state_dict['model']['language_model']['embedding']['word_embeddings']:
        param_list.append({"data": Tensor(state_dict['model']['language_model']['embedding']['word_embeddings'][key].numpy()),
                           "name": "reward_model.backbone.embedding.word_embedding.embedding_table"})

    for key in state_dict['model']['language_model']['embedding']['position_embeddings']:
        seq_len = 550
        param_list.append({"data": Tensor(state_dict['model']['language_model']['embedding']['position_embeddings'][key].numpy()[: seq_len, :]),
                           "name": "reward_model.backbone.embedding.position_embedding.embedding_table"})

    for key in state_dict['model']['language_model']['encoder']:
        if "input_layernorm" in key:
            new_key = key.replace("layers", "blocks")
            new_key = new_key.replace("input_layernorm", "layernorm1")
            new_key = new_key.replace("weight", "gamma")
            new_key = new_key.replace("bias", "beta")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].numpy().shape)
            param_list.append({"data": Tensor(state_dict['model']['language_model']['encoder'][key].numpy()),
                               "name": "reward_model.backbone."+new_key})

        if "post_attention_layernorm" in key:
            new_key = key.replace("layers", "blocks")
            new_key = new_key.replace("post_attention_layernorm", "layernorm2")
            new_key = new_key.replace("weight", "gamma")
            new_key = new_key.replace("bias", "beta")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].numpy().shape)
            param_list.append({"data": Tensor(state_dict['model']['language_model']['encoder'][key].numpy()),
                               "name": "reward_model.backbone."+new_key})
        
        if "query_key_value" in key:
            new_key = key.replace("layers", "blocks")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].numpy().shape)
            value = state_dict['model']['language_model']['encoder'][key].numpy()
            if "weight" in key:
                value = np.reshape(value, (40, 3*128, 5120))
                q, k, v = np.split(value, 3, 1)
                q = np.reshape(q, (5120, 5120))
                k = np.reshape(k, (5120, 5120))
                v = np.reshape(v, (5120, 5120))
            else:
                q, k, v = np.split(value, 3, 0)
            param_list.append({"data": Tensor(q),
                               "name": "reward_model.backbone."+new_key.replace("self_attention.query_key_value", "attention.dense1")})
            param_list.append({"data": Tensor(k),
                               "name": "reward_model.backbone."+new_key.replace("self_attention.query_key_value", "attention.dense2")})
            param_list.append({"data": Tensor(v),
                               "name": "reward_model.backbone."+new_key.replace("self_attention.query_key_value", "attention.dense3")})

        if  "self_attention.dense" in key:
            new_key = key.replace("layers", "blocks")
            new_key = new_key.replace("self_attention.dense", "attention.projection")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].numpy().shape)
            value = state_dict['model']['language_model']['encoder'][key].numpy()
            if "weight" in key:
                value = np.transpose(value, [1, 0])
            param_list.append({"data": Tensor(value),
                               "name": "reward_model.backbone."+new_key})

        if "dense_h_to_4h" in key:
            new_key = key.replace("layers", "blocks")
            new_key = new_key.replace("mlp.dense_h_to_4h", "output.mapping")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].numpy().shape)

            value = state_dict['model']['language_model']['encoder'][key].numpy()
            if "weight" in key:
                value = np.transpose(value, [1, 0])
            param_list.append({"data": Tensor(value),
                               "name": "reward_model.backbone."+new_key})
        
        if "dense_4h_to_h" in key:
            new_key = key.replace("layers", "blocks")
            new_key = new_key.replace("mlp.dense_4h_to_h", "output.projection")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].numpy().shape)

            value = state_dict['model']['language_model']['encoder'][key].numpy()
            if "weight" in key:
                value = np.transpose(value, [1, 0])
            param_list.append({"data": Tensor(value),
                               "name": "reward_model.backbone."+new_key})

        if "final_layernorm" in key:
            new_key = key.replace("weight", "gamma")
            new_key = new_key.replace("bias", "beta")
            print("new_key: ", new_key)
            print("shape: ", state_dict['model']['language_model']['encoder'][key].shape)
            param_list.append({"data": Tensor(state_dict['model']['language_model']['encoder'][key].numpy()),
                               "name": "reward_model."+new_key})
        

    for key in state_dict['model']['reward_head']:
        print("new_key: ", key)
        print("shape: ", state_dict['model']['reward_head'][key].numpy().shape)
        param_list.append({"data": Tensor(state_dict['model']['reward_head'][key].numpy()), "name": "reward_model.vHead."+key})
    
    for param in param_list:
        print(param["name"])

    save_checkpoint(param_list, mindspore_path)
    print(f"Convert finished, the output is saved to {mindspore_path}")