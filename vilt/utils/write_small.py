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
IMAGES = [
    "29333070_2190acd0246bd6a68c51b3804bd26afe-p_f.jpg",
    "29345051_c7d2badf164d077b459ab589c2044160-p_f.jpg",
    "29484993_dcfaac060ff5c6639a42259816f8f879-p_f.jpg",
    "29491324_fe91bb0a68d3e15e374d3285c2f93089-p_f.jpg",
    "29578358_3e55c71afd2094bc2cfb10eb1c4d22f8-p_f.jpg",
    "29575302_7553d648503a5dade608088ee883fcd8-p_f.jpg",
    "29557398_763eb7d555200aeed60965953dcceeb0-p_f.jpg",
    "29547375_bb898a3e816e3d335b1080ed04032de5-p_f.jpg",
    "29547333_5b1deb9c31e2d05a645ba8d9fb1c1760-p_f.jpg",
    "29515087_6d689b4b73f12f7d1dc9e5dfd6d4783d-p_f.jpg"
    ]


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
    
    
    homes = dict()
    iid = 1
    for _, home in tqdm(austin.items(), desc="Filtering Austin"):
        name = home["homeImage"].strip()
        if name in IMAGES:
            h = {"Type": home["homeType"], "DescriptionPlain": home["description"], "folder": "austin_img", "image_name": name}
            homes[iid] = h
            iid += 1
    
    # print("Overlapping ids austin:", austin_overlapping, "Overlapping NY:", ny_overlapping)
    print("Filtered the data. Length of filtered data:", len(homes))
    

    bs = []
    for ids, home in tqdm(list(homes.items()), desc="Processing small data"):
        data = path2rest(ids, f"{root}/{home['folder']}/{home['image_name']}", "train", home)
        if data:
            bs.append(data)
    
    print(f"Writing train with size {len(bs)}. This should be 10")
    write_df(bs, f"{dataset_root}/small_test.arrow")


if __name__ == "__main__":
    make_arrow("data/HousingData", "data_vilt")