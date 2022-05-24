from vilt.datasets import AmericanHomesDataset
from .datamodule_base import BaseDataModule


class AmericanHomesDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return AmericanHomesDataset

    @property
    def dataset_cls_no_false(self):
        return AmericanHomesDataset

    @property
    def dataset_name(self):
        return "amh"
