from vilt.datasets import DAGWDataset
from .datamodule_base import BaseDataModule


class DAGWDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DAGWDataset

    @property
    def dataset_name(self):
        return "dagw"
