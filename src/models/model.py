from src.models.bm25 import bm25_model


def load_model(config, data):
    if config["model"]["name"] == "bm25":
        return bm25_model(config, data)
    else:
        raise ValueError("Unknown model")
