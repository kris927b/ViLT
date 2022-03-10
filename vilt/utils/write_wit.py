import json
import pandas as pd
import pyarrow as pa
import gc
import requests
import os

from tqdm import tqdm
from glob import glob

SUB = 50000


def path2rest(path, iid2captions):
    captions, split = iid2captions[path]

    binary = requests.get(path).content

    return [
        binary,
        captions,
        "_".join(path.strip().split("/")[-3]),
        split,
    ]


def make_arrow(root, dataset_root):
    with open(f"{root}/wit_danish.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = dict()
    for cap in tqdm(captions):
        iid = cap["image_url"]
        if iid:
            iid2captions[iid] = ([cap["caption_reference_description"]], cap["split"])

    # paths = list(glob(f"{root}/images_train/*/*"))
    # random.shuffle(paths)
    # caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
    # if len(paths) == len(caption_paths):
    #     print("all images have caption annotations")
    # else:
    #     print("not all images have caption annotations")
    # print(
    #     len(paths), len(caption_paths), len(iid2captions),
    # )

    sub_len = int(len(iid2captions) // SUB)
    subs = list(range(sub_len + 1))
    for sub in subs:
        sub_paths = list(iid2captions.keys())[sub * SUB : (sub + 1) * SUB]
        bs = [path2rest(path, iid2captions) for path in tqdm(sub_paths)]
        dataframe = pd.DataFrame(
            bs,
            columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/wit_train_{sub}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()


if __name__ == "__main__":
    make_arrow("data", "data")
