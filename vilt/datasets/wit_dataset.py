from .base_dataset import BaseDataset


class WITDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        names = [f"wit_train_{i}" for i in range(9)]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="caption",
        )

    def __getitem__(self, index):
        self.get_suite(index)
