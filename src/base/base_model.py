from .base import Base
from .base_config import ConfigDict
from pytorch_lightning import LightningModule
from pytorch_lightning import data_loader
from src.utils import init_module
from src import Corpus


class BaseModel(Base, LightningModule):
    """
    a wraper of LightningModule as a base class for model system
    override functions as LightningModule needs
    """
    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        self.config = config
        self._build_model()

    @Base.log
    def init_corpus(self, corpus):
        self.corpus = corpus

    @classmethod
    def load_default_config(cls) -> ConfigDict:
        config = ConfigDict()
        return config

    @Base.log
    def _build_model(self):
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Return a list of optimizers and a list of schedulers (could be empty)
        :return:
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i):
        raise NotImplementedError

    @data_loader
    def tng_dataloader(self):
        if self.corpus is not None:
            tng_dataloader = self.corpus.build_train_dataloader()
            return tng_dataloader
        else:
            raise NotImplementedError

    @data_loader
    def val_dataloader(self):
        if self.corpus is not None:
            val_data_loader = self.corpus.build_val_dataloader()
            return val_data_loader
        else:
            return None

    @data_loader
    def test_dataloader(self):
        if self.corpus is not None:
            test_dataloader = self.corpus.build_test_dataloader()
            return test_dataloader
        else:
            return None




