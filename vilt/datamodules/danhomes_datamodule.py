from vilt.datasets import DanHomesDataset
from .datamodule_base import BaseDataModule


class DanHomesDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DanHomesDataset

    @property
    def dataset_cls_no_false(self):
        return DanHomesDataset

    @property
    def dataset_name(self):
        return "danh"
