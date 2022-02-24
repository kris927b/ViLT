from ViLT.vilt.datasets import WITDataset
from .datamodule_base import BaseDataModule


class WITDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WITDataset

    @property
    def dataset_name(self):
        return "wit"
