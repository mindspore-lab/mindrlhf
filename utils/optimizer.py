import mindspore.nn as nn


def init_optimizer(model, config):
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=config.lr,
                                   weight_decay=config.weight_decay,
                                   eps=config.eps)
    return optimizer