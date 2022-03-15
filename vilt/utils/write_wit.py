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
from base64 import decodestring
import io
import gzip

SUB = 50

LINK = "https://analytics.wikimedia.org/published/datasets/one-off/caption_competition/training/image_pixels/part-{part}-04b253b8-db8c-4d14-a23f-3433a86841b4-c000.csv.gz"


def path2rest(path, iid2captions):
    captions, split = iid2captions[path]

    binary = requests.get(path).content

    try:
        img = Image.open(binary).tobytes()
    except FileNotFoundError:
        img = None
    except ValueError:
        print(binary)
        img = None

    return img


def find_images(iid2captions, pixels):
    for line in pixels:
        l = line.decode().strip().split("\t")
        if l[0] in iid2captions.keys():
            img = Image.frombytes("RGB", (300, 300), decodestring(l[1])).tobytes()
            yield [
                img,
                iid2captions[l[0]][0],
                "_".join(l[0].strip().split("/")[-3]),
                iid2captions[l[0]][1],
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

    sub_len = int(200 // SUB)
    subs = list(range(sub_len + 1))
    for sub in subs:
        sub_paths = list(range(sub * SUB, (sub + 1) * SUB))
        bs = [
            find_images(
                iid2captions,
                gzip.open(
                    io.BytesIO(
                        requests.get(
                            LINK.format(part="0" * (5 - len(str(i))) + str(i))
                        ).content
                    ),
                    mode="rb",
                ),
            )
            for i in tqdm(sub_paths)
        ]
        dataframe = pd.DataFrame(
            bs,
            columns=["image", "caption", "image_id", "split"],
        )
        print("Instances:", len(dataframe))
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
    make_arrow("data", "data2")
