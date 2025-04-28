from data_loader.gsm8k_loader import GSM8KLoader
from data_loader.sciq_loader import SciQLoader


def get_dataloader(dataset: str):
    # return dataloader based on the dataset name
    if dataset == "GSM8K":
        return GSM8KLoader()
    elif dataset == "SciQ":
        return SciQLoader()
    raise Exception("In get_dataloader(), dataset is not not detected, should be one of: [GSM8K, SciQ]!")
