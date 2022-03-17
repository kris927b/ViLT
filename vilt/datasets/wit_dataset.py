from .base_dataset import BaseDataset


class WITDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"wit_train_{i}" for i in range(3)]
        elif split == "val":
            names = ["wit_train_0"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="caption",
        )

    def __getitem__(self, index):
        return self.get_suite(index)
