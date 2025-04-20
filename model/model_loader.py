from model.base_model import BaseModel
from model.local_model import LocalModel
from model.openai_model import OpenAIModel

def load_model(model_type: str, model: str) -> BaseModel:
    if model_type == "openai":
        return OpenAIModel(model)
    elif model_type == "local":
        return LocalModel(model)
    else:
        raise Exception("In util.py load_model(), model_type is not recognized.")