from src.models.bm25 import bm25_model
from src.models.bert import BERT_Model


def load_model(config, data):
    if config["model"]["name"] == "bm25":
        return bm25_model(config, data)
    elif config["model"]["name"] == "bert":
        return BERT_Model(config)
    else:
        raise ValueError("Unknown model")
