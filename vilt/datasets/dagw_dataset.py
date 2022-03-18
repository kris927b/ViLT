from .base_dataset import BaseDataset


class DAGWDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"], f"Split: {split}"
        if split == "test":
            split = "val"

        if split == "train":
            names = [f"dagw_train_{i}" for i in range(3)]
        elif split == "val":
            names = ["dagw_train_0"]

        super().__init__(
            *args, **kwargs, names=names, text_column_name="text", text_only=True
        )

    def __getitem__(self, index):
        return self.get_suite(index)
