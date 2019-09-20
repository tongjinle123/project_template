from src.base import ConfigDict


def init_module(Module, module_name, change_args=None):
    if module_name is None:
        print(f'None for {Module.__name__}.')
        return None
    module = getattr(module_name, Module)
    config = module.load_default_config()
    if change_args is not None:
        change_config = ConfigDict()
        change_config.update(change_args)
        config.combine(change_config)
    module = module(config)
    return module
