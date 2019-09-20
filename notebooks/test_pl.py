import pytorch_lightning as pl
import test_tube
import torch as t
from torch.utils.data import DataLoader, Dataset
import random
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser


#%%
class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        x1 = [random.randint(1, 100) for i in range(10000)]
        x2 = [random.randint(1, 100) for i in range(10000)]

        y = [a+b for a, b in zip(x1, x2)]
        self.data = [(i, j, k) for i, j, k in zip(x1, x2, y)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

dataset = TestDataset()
dataloader = DataLoader(dataset, batch_size=3)
#%%
class TestModel(pl.LightningModule):
    def __init__(self):
        super(TestModel, self).__init__()


    @dataloader
    def tng_dataloader(self):
        return DataLoader(TestDataset())


#%%
class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.data = [1,2,3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

dataset = TestDataset()
dataloader = DataLoader(dataset, batch_size=3)

class TestDataset2(Dataset):
    def __init__(self):
        super(TestDataset2, self).__init__()
        self.data = [4,5,6]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

dataset2 = TestDataset2()
dataloader2 = DataLoader(dataset2, batch_size=3)

a = t.utils.data.ConcatDataset([dataloader, dataloader2])
for i in a:
    print(i)
#%%

