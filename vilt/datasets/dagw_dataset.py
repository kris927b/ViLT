from .base_dataset import BaseDataset


class DAGWDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"

        if split == "train":
            names = ["dagw_train"]
        elif split == "val":
            names = ["dagw_val"]

        super().__init__(
            *args, **kwargs, names=names, text_column_name="text", text_only=True
        )

    def __getitem__(self, index):
        self.get_suite(index)
