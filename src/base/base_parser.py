from .base import Base
from .base_config import ConfigDict
from src.utils import Vocab


class BaseParser(Base):
    """
    base class for parser,
    """
    def __init__(self, config):
        super(BaseParser, self).__init__(config)
        self._init_vocab()

    @classmethod
    def load_default_config(cls) -> ConfigDict:
        config = ConfigDict(
            vocab_path=None
        )
        return config

    @Base.log
    def _init_vocab(self):
        self.vocab = Vocab()
        self.vocab.load(self.config.vocab_path)

    def parse_train(self):
        pass

    def parse_test(self):
        pass

    def parse_predump(self):
        pass

    def parse_predump_train(self):
        pass
