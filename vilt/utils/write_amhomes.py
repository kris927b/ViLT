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

AUSTIN = ["homeImage", "description", "homeType"]
NEWYORK = ["photos/0", "description", "resoFactsStats/homeType"] # Remember to split on / and take the last one to get image name


def path2rest(iid, path, split, annotations):
    # iid = os.path.basename(path).split(".")[0]

    with open(path, "rb") as fp:
        binary = fp.read()

    caption = sent_tokenize(annotations["DescriptionPlain"], language='english')
    if len(caption) == 0:
        return None

    property_type = LABEL2ANS[annotations["Type"]]

    return [binary, [caption[0]], property_type, iid, split]


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
    with open(f"{root}/austinHousingData.json", "r") as fp:
        austin = json.load(fp)
    with open(f"{root}/newyork_housing.json", "r") as fp:
        newyork = json.load(fp)
    print("Loaded the data")
    
    
    homes = dict()
    iid = 1
    for _, home in tqdm(austin.items(), desc="Filtering Austin"):
        if home["homeType"] in LABEL2ANS.keys():
            name = home["homeImage"].strip()
            h = {"Type": home["homeType"], "DescriptionPlain": home["description"], "folder": "austin_img", "image_name": name}
            homes[iid] = h
            iid += 1

    for ids, home in tqdm(newyork.items(), desc="Filtering New York"):
        if home["resoFactsStats/homeType"] in LABEL2ANS.keys():
            name = os.path.basename(home["imgpath"])
            h = {"Type": home["resoFactsStats/homeType"], "DescriptionPlain": home["description"], "folder": "ny_img", "image_name": name}
            homes[iid] = h
            iid += 1
    
    # print("Overlapping ids austin:", austin_overlapping, "Overlapping NY:", ny_overlapping)
    print("Filtered the data. Length of filtered data:", len(homes))
    
    train_size = int(len(homes) * 0.8)
    val_size = int(len(homes) * 0.1)
    test_size = int(len(homes) * 0.1)

    bs = []
    for ids, home in tqdm(list(homes.items())[:train_size], desc="Processing Train"):
        data = path2rest(ids, f"{root}/{home['folder']}/{home['image_name']}", "train", home)
        if data:
            bs.append(data)
    
    print(f"Writing train with size {len(bs)}. This should be {train_size}")
    write_df(bs, f"{dataset_root}/amhomes_train.arrow")
    
    bs = []
    for ids, home in tqdm(list(homes.items())[train_size:train_size+val_size], desc="Processing Val"):
        data = path2rest(ids, f"{root}/{home['folder']}/{home['image_name']}", "val", home)
        if data:
            bs.append(data)
    
    print(f"Writing val with size {len(bs)}. This should be {val_size}")
    write_df(bs, f"{dataset_root}/amhomes_val.arrow")

    bs = []
    for ids, home in tqdm(list(homes.items())[train_size+val_size:], desc="Processing Test"):
        data = path2rest(ids, f"{root}/{home['folder']}/{home['image_name']}", "test", home)
        if data:
            bs.append(data)
    
    print(f"Writing test with size {len(bs)}. This should be {test_size}")
    write_df(bs, f"{dataset_root}/amhomes_test.arrow")


if __name__ == "__main__":
    make_arrow("data/HousingData", "data_vilt")