import re


def clean_string(input_string):
    """
    Cleans the input string by removing \n and *.
    """
    return re.sub(r'[\n|*]', '', input_string)


def cuta_last_appearance(txt: str, keyword: str) -> str:
    """
    Cut the txt after the last appearance of the keyword.
    """
    result_start = txt.rfind(keyword)
    if result_start < 0:
        raise Exception("In retrieve_llama_summarized_content(), there should be a keyword but there isn't")
    result_start += len(keyword)
    result = txt[result_start:].strip()
    return result
