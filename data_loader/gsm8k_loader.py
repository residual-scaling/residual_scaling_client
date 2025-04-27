from data_loader.base_loader import BaseLoader
from config import data_path
from util import cut_last_appearance, save_utial_last_appearance
import json



def format_gsm8k_item(raw_item):
    """
    Format the raw data from GSM8K to the standard format.
    """
    result = {}
    result["question"] = raw_item['question']
    result["truth"] = cut_last_appearance(raw_item['answer'], "####").strip()
    result["options"] = []
    result["reasoning"] = save_utial_last_appearance(raw_item['answer'], "####").strip()
    return result


def load_gsm8k_json(file_name: str):
    file_path = data_path["gsm8k"] + file_name
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            data = format_gsm8k_item(json.loads(line))
            result.append(data)
    return result


class GSM8KLoader(BaseLoader):
    def __init__(self):
        self.data = []
        self.data += load_gsm8k_json("train.jsonl")
        self.data += load_gsm8k_json("test.jsonl")
        self.size = len(self.data)
