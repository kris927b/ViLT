from importlib.resources import contents
import json
import pandas as pd
import pyarrow as pa
import gc
import requests
import os

from tqdm import tqdm
from glob import glob
from PIL import Image
from base64 import b64decode
import io
import gzip

SUB = 50000


def make_arrow(root, dataset_root):
    with open(f"{root}/dagw.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = dict()
    for iid, cap in tqdm(enumerate(captions)):
        if iid:
            iid2captions[iid] = ([cap], "train")

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
        bs = [[iid, iid2captions[iid][0], iid2captions[iid][1]] for iid in sub_paths ]
        dataframe = pd.DataFrame(
            bs,
            columns=["id", "text", "split"],
        )
        
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/dagw_train_{sub}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()


if __name__ == "__main__":
    make_arrow("data", "data2")
