from data_loader.base_loader import BaseLoader
from config import data_path
import pandas as pd
import random


def format_sciq_item(raw_item, shuffle_options: bool = False):
    """
    Format the raw data from SciQ to the standard format.
    """
    result = {}
    result["question"] = raw_item["question"]
    result["truth"] = raw_item["correct_answer"]
    result["options"] = [
        raw_item["distractor1"],
        raw_item["distractor2"],
        raw_item["distractor3"],
        raw_item["correct_answer"]
    ]
    if shuffle_options:
        random.shuffle(result["options"])
    result["reasoning"] = raw_item["support"]
    return result


def load_sciq_json(file_name: str, shuffle_options: bool = False):
    file_path = data_path["sciq"] + file_name
    result = []
    df = pd.read_parquet(file_path)
    for _, row in df.iterrows():
        data = format_sciq_item(row, shuffle_options)
        result.append(data)
    return result



class SciQLoader(BaseLoader):
    def __init__(self):
        self.data = []
        self.data += load_sciq_json("train-00000-of-00001.parquet")
        self.data += load_sciq_json("test-00000-of-00001.parquet")
        self.data += load_sciq_json("validation-00000-of-00001.parquet")
        self.size = len(self.data)
