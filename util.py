from data_loader.gsm8k_loader import GSM8KLoader
from data_loader.sciq_loader import SciQLoader
import re


def clean_string(input_string):
    """
    Cleans the input string by removing \n and *.
    """
    return re.sub(r'[\n|*|\'|\"]', '', input_string.strip()).strip()

def cut_last_appearance(txt: str, keyword: str) -> str:
    """
    Cut the txt after the last appearance of the keyword.
    """
    result_start = txt.rfind(keyword)
    if result_start < 0:
        raise Exception("In retrieve_llama_summarized_content(), there should be a keyword but there isn't")
    result_start += len(keyword)
    result = txt[result_start:].strip()
    return result


def save_utial_last_appearance(txt: str, keyword: str) -> str:
    """
    Save the txt after the last appearance of the keyword.
    """
    result_start = txt.rfind(keyword)
    if result_start < 0:
        raise Exception("In retrieve_llama_summarized_content(), there should be a keyword but there isn't")
    result = txt[:result_start].strip()
    return result


def log_flush(fileIO, txt: str):
    """
    Write and flush to the disk.
    """
    fileIO.write(txt)
    fileIO.write("\n")
    fileIO.flush()


def get_generated_text(generation_result) -> str:
    return generation_result[0]['generated_text']


def get_dataloader(dataset: str):
    # return dataloader based on the dataset name
    if dataset == "GSM8K":
        return GSM8KLoader()
    elif dataset == "SciQ":
        return SciQLoader()
    raise Exception("In get_dataloader(), dataset is not not detected, should be one of: [GSM8K, SciQ]!")
