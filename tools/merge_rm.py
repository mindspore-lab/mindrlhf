import torch
import os

orig_model_dir = "/path1/"
new_model_dir = "path2"

rank_names = ['mp_rank_00', 'mp_rank_01', 'mp_rank_02', 'mp_rank_03', 'mp_rank_04', 'mp_rank_05', 'mp_rank_06',
              'mp_rank_07']

# adjacent ranks should be concatenated
current_rank_dir = os.path.join(orig_model_dir, rank_names[0])
current_rank_dir1 = os.path.join(orig_model_dir, rank_names[1])
current_rank_dir2 = os.path.join(orig_model_dir, rank_names[2])
current_rank_dir3 = os.path.join(orig_model_dir, rank_names[3])
current_rank_dir4 = os.path.join(orig_model_dir, rank_names[4])
current_rank_dir5 = os.path.join(orig_model_dir, rank_names[5])
current_rank_dir6 = os.path.join(orig_model_dir, rank_names[6])
current_rank_dir7 = os.path.join(orig_model_dir, rank_names[7])

current_rank_dir = []
parts = []
for i in range(8):
    current_rank_dir.append(os.path.join(orig_model_dir, rank_names[i]))
    parts.append(torch.load(os.path.join(current_rank_dir[i], 'model_optim_rng.pt'), map_location='cpu'))


parts[0]['model']['language_model']['embedding']['word_embeddings']['weight'] \
    = torch.cat((
        parts[0]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[1]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[2]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[3]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[4]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[5]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[6]['model']['language_model']['embedding']['word_embeddings']['weight'],
        parts[7]['model']['language_model']['embedding']['word_embeddings']['weight']), 0)

###################### merge langauge model #############################
for j in range(0, 39):
    # mrege kqv
    kqv_key_dim_dict = {
        'self_attention.query_key_value.weight': 0, 
        'self_attention.query_key_value.bias': 0}
    for key, dim in kqv_key_dim_dict.items():
        parts[0]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)] \
            = torch.cat((
                parts[0]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[1]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[2]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[3]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[4]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[5]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[6]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[7]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)]), dim)

    # merge other parts in attn
    other_key_dim_dict = {'self_attention.dense.weight': 1}
    for key, dim in other_key_dim_dict.items():
        parts[0]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)] \
            = torch.cat((
                parts[0]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[1]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[2]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[3]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[4]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[5]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[6]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[7]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)]), dim)

    # merge ffn
    ffn_key_dim_dict = {
        'mlp.dense_h_to_4h.bias': 0,
        'mlp.dense_h_to_4h.weight': 0,
        'mlp.dense_4h_to_h.weight': 1}
    for key, dim in ffn_key_dim_dict.items():
        parts[0]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)] \
            = torch.cat((
                parts[0]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[1]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[2]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[3]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[4]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[5]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[6]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)],
                parts[7]['model']['language_model']['encoder']['layers.{0}.{1}'.format(j, key)]), dim)
    
new_model = {key: parts[0][key] for key in ['model', 'checkpoint_version']}

if not os.path.exists(new_model_dir):
    os.makedirs(new_model_dir)

torch.save(new_model, os.path.join(new_model_dir, 'rm_model.pt'))

