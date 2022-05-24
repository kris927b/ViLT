import json
import pandas as pd
import pyarrow as pa
import re
import os
import gc

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter


LABEL2ANS = {
    "Single Family": 0,
    "Condo": 1,
    "Multiple Occupancy": 2,
    "Vacant Land": 3,
    "Townhouse": 4,
    "Recreational": 5,
    "Country House": 6,
    "Villa": 0,
    "Ejerlejlighed": 1,
    "Kollektiv": 2,
    "Helårsgrund": 3,
    "Fritidsgrund": 3,
    "Rækkehus": 4,
    "Fritidsbolig": 5,
    "Landejendom": 6
}


def path2rest(path, split, annotations):
    iid = os.path.basename(path).split(".")[0]

    with open(path, "rb") as fp:
        binary = fp.read()

    caption = sent_tokenize(annotations["DescriptionPlain"], language='danish')[0]

    property_type = LABEL2ANS[annotations["Type"]]

    return [binary, [caption], property_type, iid, split]


def write_df(bs, path):
    dataframe = pd.DataFrame(
        bs,
        columns=["image", "caption", "type", "image_id", "split"],
    )
    
    table = pa.Table.from_pandas(dataframe)
    # os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    del dataframe
    del table
    del bs


def make_arrow(root, dataset_root):
    with open(f"{root}/danish_homes.json", "r") as fp:
        captions = json.load(fp)
    print("Loaded the data")
    homes = dict()

    for ids, home in captions.items():
        if home["Type"] in LABEL2ANS.keys():
            homes[ids] = home
    print("Filtered the data")
    
    train_size = int(len(homes) * 0.8)
    val_size = int(len(homes) * 0.1)
    test_size = int(len(homes) * 0.1)

    bs = []
    for ids, home in tqdm(list(homes.items())[:train_size]):
        
        data = path2rest(f"{root}/dk_img/{ids}.jpg", "train", home)
        bs.append(data)
    
    print(f"Writing train with size {len(bs)}. This should be {train_size}")
    write_df(bs, f"{dataset_root}/danhomes_train.arrow")
    
    bs = []
    for ids, home in tqdm(list(homes.items())[train_size:train_size+val_size]):
        
        data = path2rest(f"{root}/dk_img/{ids}.jpg", "val", home)
        bs.append(data)
    
    print(f"Writing val with size {len(bs)}. This should be {val_size}")
    write_df(bs, f"{dataset_root}/danhomes_val.arrow")

    bs = []
    for ids, home in tqdm(list(homes.items())[train_size+val_size:]):
        
        data = path2rest(f"{root}/dk_img/{ids}.jpg", "test", home)
        bs.append(data)
    
    print(f"Writing test with size {len(bs)}. This should be {test_size}")
    write_df(bs, f"{dataset_root}/danhomes_test.arrow")


if __name__ == "__main__":
    make_arrow("data/DanHomes", "data_vilt")