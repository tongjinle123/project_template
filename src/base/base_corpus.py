from .base_config import ConfigDict
from .base import Base
import torch as t
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from src.utils import init_module


class BaseCorpus(Base):
    """
    base class for corpus which is used for:
        1.building dataloader for model
        2.transforming data for inference
    """
    def __init__(self, config):
        super(BaseCorpus, self).__init__(config)

    @classmethod
    def load_default_config(cls) -> ConfigDict:
        config = ConfigDict()
        return config

    @Base.log
    def init_parser(self, parser):
        self.parser = parser

    @Base.log
    def build_tng_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @Base.log
    def build_val_dataloader(self):
        return None

    @Base.log
    def build_test_dataloader(self):
        return None

