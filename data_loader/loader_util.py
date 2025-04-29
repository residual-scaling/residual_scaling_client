from data_loader.gsm8k_loader import GSM8KLoader
from data_loader.sciq_loader import SciQLoader
from data_loader.mmlu_loader import MMLULoader


def get_dataloader(dataset: str, subject: str = "high_school_biology"):
    # return dataloader based on the dataset name
    if dataset == "GSM8K":
        return GSM8KLoader()
    elif dataset == "SciQ":
        return SciQLoader()
    elif dataset == "MMLU":
        return MMLULoader(subject)
    raise Exception("In get_dataloader(), dataset is not not detected, should be one of: [GSM8K, SciQ, MMLU]!")
