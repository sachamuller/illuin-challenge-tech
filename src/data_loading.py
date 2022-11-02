import json
import os
import ast
from typing import Dict
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def load_squad_to_df(config: Dict):
    if config["data_loading"]["evaluate_on_train_or_test"] == "train":
        data_path = config["data_loading"]["train_path"]
    elif config["data_loading"]["evaluate_on_train_or_test"] == "test":
        data_path = config["data_loading"]["test_path"]
    else:
        raise ValueError(
            f"Parameter data_loading.evaluate_on_train_or_test in config should be 'train' or 'test', \
            got {config['data_loading']['evaluate_on_train_or_test']}"
        )
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

    if "is_impossible" in data_df.columns:  # means we are dealing with the 2.0 dataset
        if drop_impossible:
            data_df = data_df[~data_df["is_impossible"]]
        else:
            data_df[data_df["is_impossible"]]["answers"] = data_df[
                data_df["is_impossible"]
            ]["plausible_answers"]
        data_df = data_df.drop(columns=["plausible_answers"])

    # A few contexts appear twice in 'paragraphs', with different questions,
    # this is why we group by context instead of assigning an id at paragraphs level
    # and repeating this id
    data_df["context_id"] = data_df.groupby("context", sort=False).ngroup()
    data_df = data_df.reset_index()
    return data_df


class SquadContexts(Dataset):
    def __init__(self, data_df):

        self.contexts = data_df["context"].unique()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx]

    def truncate_beginning(self, start_index):
        self.contexts = self.contexts[start_index:]


class SquadQuestions(Dataset):
    def __init__(self, config, data_df):
        self.full_df = data_df

        self.random_state = config["seed"]
        self.sample_df = self.full_df

    def __len__(self):
        return len(self.sample_df.index)

    def __getitem__(self, idx):
        return self.sample_df.iloc[idx].name, self.sample_df["question"].iloc[idx]

    def reduce_to_sample(self, frac, previous_scores=None):
        if previous_scores is None:
            self.sample_df = self.full_df.sample(
                frac=frac, random_state=self.random_state
            )
        else:
            questions_idx_with_computed_score = list(
                previous_scores.sum(dim=1).nonzero().squeeze().numpy()
            )
            questions_idx_with_NO_computed_score = [
                idx
                for idx in self.full_df.index
                if idx not in questions_idx_with_computed_score
            ]
            self.sample_df = self.full_df.loc[
                questions_idx_with_NO_computed_score
            ].sample(frac=frac, random_state=self.random_state)


if __name__ == "__main__":

    # Example of use
    import yaml

    config = yaml.safe_load(open("examples/config.yaml"))
    data_df = load_squad_to_df(config)
    print(data_df)
