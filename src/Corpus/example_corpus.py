from src.base import BaseCorpus
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class ExampleCorpus(BaseCorpus):

    def __init__(self, config):
        super(ExampleCorpus, self).__init__(config)

    def build_tng_dataloader(self) -> DataLoader:
        pass

    def build_test_dataloader(self) -> DataLoader:
        pass

    def build_val_dataloader(self) -> DataLoader:
        pass

