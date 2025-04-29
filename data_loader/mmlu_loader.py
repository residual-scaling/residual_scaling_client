from data_loader.base_loader import BaseLoader
from config import data_path
import pandas as pd
import random


def format_mmlu_item(raw_item, shuffle_options: bool = False):
    """
    Format the raw data from SciQ to the standard format.
    """
    result = {}
    result["question"] = raw_item["question"]
    result["truth"] = raw_item["choices"][raw_item["answer"]]
    result["options"] = raw_item["choices"]
    if shuffle_options:
        random.shuffle(result["options"])
    result["reasoning"] = ""
    return result


def load_mmlu_json(subject: str, file_name: str, shuffle_options: bool = False):
    file_path = data_path["mmlu"] + subject + "/" + file_name
    result = []
    df = pd.read_parquet(file_path)
    for _, row in df.iterrows():
        data = format_mmlu_item(row, shuffle_options)
        result.append(data)
    return result


class MMLULoader(BaseLoader):
    def __init__(self, subject: str = "high_school_biology"):
        self.data = []
        self.data += load_mmlu_json(subject, "dev-00000-of-00001.parquet")
        self.data += load_mmlu_json(subject, "test-00000-of-00001.parquet")
        self.data += load_mmlu_json(subject, "validation-00000-of-00001.parquet")
        self.size = len(self.data)