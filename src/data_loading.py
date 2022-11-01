import json
import torch
from typing import Dict
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


def load_squad_to_df(config: Dict, test: bool = False):
    if not test:
        data_path = config["data_loading"]["train_path"]
    else:
        data_path = config["data_loading"]["test_path"]
    data_json = json.load(open(data_path))["data"]
    df = json_to_dataframe(data_json, config["data_loading"]["drop_impossible"])
    return df


def json_to_dataframe(data_json: Dict, drop_impossible: bool):
    data_level = pd.json_normalize(data_json)  # contains title and paragraphs
    repeated_titles = np.repeat(
        data_level["title"].to_numpy(), data_level["paragraphs"].apply(len).to_numpy()
    )

    paragraphs_level = pd.json_normalize(
        data_json, ["paragraphs"]
    )  # contains qas and context
    paragraphs_level["title"] = repeated_titles
    repeated_context = np.repeat(
        paragraphs_level["context"].to_numpy(),
        paragraphs_level["qas"].apply(len).to_numpy(),
    )
    repeated_twice_titles = np.repeat(
        paragraphs_level["title"].to_numpy(),
        paragraphs_level["qas"].apply(len).to_numpy(),
    )

    data_df = pd.json_normalize(
        data_json, ["paragraphs", "qas"]
    )  # contains question, id, answers, is_impossible and plausible_answers
    data_df["title"] = repeated_twice_titles
    data_df["context"] = repeated_context

    if drop_impossible:
        data_df = data_df[~data_df["is_impossible"]]
    else:
        data_df[data_df["is_impossible"]]["answers"] = data_df[
            data_df["is_impossible"]
        ]["plausible_answers"]
    data_df = data_df.drop(columns=["plausible_answers"])
    return data_df


class SquadContexts(Dataset):
    def __init__(self, config, test=False):
        data_df = load_squad_to_df(config, test)

        self.contexts = data_df["context"].unique()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx]


if __name__ == "__main__":

    # Example of use
    import yaml

    config = yaml.safe_load(open("examples/config.yaml"))
    data_df = load_squad_to_df(config)
    print(data_df)
