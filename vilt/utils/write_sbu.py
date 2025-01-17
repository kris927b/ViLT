import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os
import requests
from io import BytesIO

from tqdm import tqdm
from glob import glob
from PIL import Image, UnidentifiedImageError

SUB = 10000


def path2rest(path, iid2captions):
    name = path.split("/")[-1].split(".")[0]
    split = "train"
    iid = path
    t = path.split(".")[-1]
    captions = iid2captions[iid]
    binary = requests.get(path).content
    
    try:
        img = Image.open(BytesIO(binary))
    except UnidentifiedImageError:
        return None
    
    b = BytesIO()
    img.convert('RGB').save(b, 'PNG')

    return [
        b.getvalue(),
        captions,
        name,
        split,
    ]


def make_arrow(root, dataset_root):
    with open(f"{root}/annot.json", "r") as fp:
        captions = json.load(fp)

    iid2captions = dict()
    for cap in tqdm(captions):
        iid = cap[0]
        if iid:
            iid2captions[iid] = [cap[1]]

    # paths = list(glob(f"{root}/images_train/*/*"))
    # random.shuffle(paths)
    # caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
    # if len(paths) == len(caption_paths):
    #     print("all images have caption annotations")
    # else:
    #     print("not all images have caption annotations")
    # print(
    #     len(paths),
    #     len(caption_paths),
    #     len(iid2captions),
    # )

    sub_len = int(len(iid2captions) // SUB)
    subs = list(range(sub_len + 1))
    for sub in subs:
        sub_paths = list(iid2captions.keys())[sub * SUB : (sub + 1) * SUB]
        bs = [path2rest(path, iid2captions) for path in tqdm(sub_paths)]
        bs = list(filter(None, bs))
        print(len(bs))
        dataframe = pd.DataFrame(
            bs,
            columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/sbu_{sub}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()


if __name__ == "__main__":
    make_arrow("data", "data2")
