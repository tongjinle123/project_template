from .base_config import ConfigDict
import time


class Base:
    """
    base class for all
    """
    def __init__(self, config):
        self.config = config

    @classmethod
    def log(self, func):
        def wrapper(*args, **kwargs):
            print(f'executing: {func.__name__}\n')
            func(*args, **kwargs)
        return wrapper

    @classmethod
    def load_default_config(cls) -> ConfigDict:
        """
        override this func adding args for the children class
        :return: config: ConfigDict
        """
        config = ConfigDict()
        # config.add(
        #
        # )
        return config
