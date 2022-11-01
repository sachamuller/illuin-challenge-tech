import json
import os
import ast
from typing import Dict
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


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
    repeated_context_id = np.repeat(
        np.arange(len(paragraphs_level.index)),
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
    data_df["context_id"] = repeated_context_id

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

    def truncate_beginning(self, start_index):
        self.contexts = self.contexts[start_index:]


class SquadQuestions(Dataset):
    def __init__(self, config, test=False):
        self.save_path = config["model_parameters"]["bert"]["prediction_df_path"]

        if os.path.exists(self.save_path):
            self.full_df = pd.read_csv(
                self.save_path,
            )
            self.full_df["scores"] = self.full_df["scores"].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
        else:
            data_df = load_squad_to_df(config, test)
            self.full_df = data_df[["question", "context", "context_id"]]
            self.full_df["scores"] = np.nan
            self.full_df["scores"] = self.full_df["scores"].astype("object")

        self.random_state = config["seed"]

        self.sample_df = self.full_df

    def __len__(self):
        return len(self.sample_df.index)

    def __getitem__(self, idx):
        return self.sample_df["question"].iloc[idx]

    def reduce_to_sample(self, frac, new_samples=True):
        if new_samples:
            self.sample_df = self.full_df[self.full_df["scores"].isna()].sample(
                frac=frac, random_state=self.random_state
            )
        else:
            self.sample_df = self.full_df.sample(
                frac=frac, random_state=self.random_state
            )

    def save_batch_scores(self, questions_batch, scores):
        for i, question in enumerate(questions_batch):
            question_idx = self.full_df.index[
                self.full_df["question"] == question
            ].item()
            self.full_df.at[question_idx, "scores"] = list(scores[i].numpy())

        self.full_df.to_csv(self.save_path, index=False)


if __name__ == "__main__":

    # Example of use
    import yaml

    config = yaml.safe_load(open("examples/config.yaml"))
    data_df = load_squad_to_df(config)
    print(data_df)
