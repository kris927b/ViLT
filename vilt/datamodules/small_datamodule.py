from vilt.datasets import SmallHomesDataset
from .datamodule_base import BaseDataModule


class SmallHomesDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SmallHomesDataset

    @property
    def dataset_cls_no_false(self):
        return SmallHomesDataset

    @property
    def dataset_name(self):
        return "smallh"
