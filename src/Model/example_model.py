from src.base import BaseModel
from .utils import WarmUpScheduler


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config=config)

    def _build_model(self):
        pass