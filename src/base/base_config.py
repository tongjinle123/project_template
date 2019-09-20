import yaml


class ConfigDict:
    """
    base class for config
    usages :
    config = ConfigDict(b='b')
    config.add(a='a')
    config.update(kwargs)
    config.combine(config)
    """
    def __init__(self, **kwargs):
        self.config = {}
        for key, value in kwargs.items():
            self.config[key] = value

    def __getattr__(self, item):
        return self.config[item]

    def add(self, *args, **kwargs):
        """
        add configs
        :param args:
        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            self.config[key] = value
        if args != ():
            for key, value in args[0].items():
                self.config[key] = value

    def update(self, *args, **kwargs):
        """
        update configs
        :param args:
        :param kwargs:
        :return:
        """
        for key, value in kwargs.items():
            assert key in self.config
        self.config.update(kwargs)
        if args != ():
            for key, value in args[0].items():
                assert key in self.config
                self.config[key] = value

    def combine(self, other):
        """
        combine other config with this config，the 'other config' shall prevail
        :param other: ConfigDict
        :return: None
        """
        self.add(other.config)

    @property
    def show(self):
        """
        show config args
        :return: None
        """
        print(f'config:')
        for key, value in self.config.items():
            print(f'\t{key}: {value}')

    def save(self, path):
        yaml.dump(self.config, open(path, 'w'))
        print(f'config saved to {path}')

    def load(self, path):
        self.config = yaml.safe_load(open(path))
        print(f'config loaded from {path}')
