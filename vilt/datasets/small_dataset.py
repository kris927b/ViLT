from .base_dataset import BaseDataset


class SmallHomesDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]

        names = [f"small_val"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="caption",
        )

    def __getitem__(self, index):
        ret = self.get_suite(index)
        index, _ = self.index_mapper[index]
        answers = self.table["type"][index].as_py()

        ret["label"] = answers

        return ret