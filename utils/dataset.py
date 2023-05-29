import mindspore
import numpy as np
from mindspore import ParameterTuple, nn, jit, context
from mindspore.common import Parameter, Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import MindDataset

class IteratorDataset:
    def __init__(self, data):
        self._index = 0
        self.length = len(data)
        self.data = data

    def __next__(self):
        if self._index >= self.length:
            raise StopIteration
        else:
            item = (self.data[self._index].query_tensor,
                    self.data[self._index].sample_tensor,
                    self.data[self._index].logprobs,
                    self.data[self._index].values,
                    self.data[self._index].rewards
                    )
            self._index += 1
            return item

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return self.length

class ReplayBuffer(nn.Cell):
    def __init__(self, batch_size, capacity, shapes, dtypes, seed=0):
        super(ReplayBuffer, self).__init__()
        items = []
        for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
            shape = (capacity,) + shape
            name = "buffer_" + str(i)
            items.append(Parameter(Tensor(np.zeros(shape), dtype), name=name, requires_grad=False))
        self.items = ParameterTuple(items)
        self.count = Parameter(0, name="count", requires_grad=False)

        self.hyper_map = C.HyperMap()

        self.arange = Tensor(np.arange(capacity).reshape(-1, 1), mindspore.int32)
        self.sampler = P.UniformCandidateSampler(1, batch_size, True, capacity, seed, False)

    def update(self, indice, data, value):
        # indice = indice + Tensor(np.arange(data.shape[0]))
        data[indice] = value
        return data

    def push(self, values):
        self.hyper_map(F.partial(self.update, self.count), self.items, values)
        self.count += 1
        return self.count

    def gather(self, indices, data):
        return P.Gather()(data, indices, 0)

    def sample(self):
        indices, _, _ = self.sampler(self.arange)
        return self.hyper_map(F.partial(self.gather, indices), self.items)

    def clear(self):
        self.count = 0

def create_ppo_dataset(data, config):
    pipeline = IteratorDataset(data)
    dataset = GeneratorDataset(pipeline, column_names=config.column_names)
    dataset = dataset.batch(batch_size=config.batch_size)
    return dataset

def create_experience_dataset(config):
    columns_to_project = ["prompt_ids", "prompt_mask", "original_sample_ids", "original_sample_mask"]
    prompt_dataloader = MindDataset(config.train_dataset_dir).project(columns=columns_to_project)
    prompt_dataloader = prompt_dataloader.batch(batch_size=config.chunk_size, drop_remainder=True)
    train_iterator = prompt_dataloader.create_tuple_iterator()
    val_dataloader = MindDataset(config.val_dataset_dir).project(columns=columns_to_project)
    val_dataloader = val_dataloader.batch(batch_size=config.chunk_size, drop_remainder=True)
    val_iterator = val_dataloader.create_tuple_iterator()
    return train_iterator, val_iterator

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    buffer = ReplayBuffer(8, 128, [(500,), (50,)], [mindspore.int32, mindspore.float32])

    @jit
    def test_graph():
        for i in range(128):
            buffer.push((Tensor(np.ones(500,) * i), Tensor(np.ones(50,) * i)))

        return buffer.sample()

    samples = test_graph()
    assert len(samples) == 2
    assert (samples[0][:, 0] == samples[1][:, 0]).all().asnumpy()
