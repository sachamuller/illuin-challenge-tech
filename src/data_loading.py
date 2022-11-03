import json
from typing import Union

import numpy as np
import pandas as pd
import torch

pd.options.mode.chained_assignment = None


def load_squad_to_df(config: dict) -> pd.DataFrame:
    """Returns the SQuAD dataset (train or test according to config),
    as a pandas dataframe.

    Args:
        config (dict): config dictionnary coming from yaml file

    Raises:
        ValueError: if data_loading.train_or_test in config has a value
        other than 'train' or 'test'

    Returns:
        pd.DataFrame: DataFrame containing all the contexts, questions and
        answers. The columns are : 'answers', 'question', 'id', 'title',
        'context' and 'context_id'
    """
    if config["data_loading"]["train_or_test"] == "train":
        data_path = config["data_loading"]["train_path"]
    elif config["data_loading"]["train_or_test"] == "test":
        data_path = config["data_loading"]["test_path"]
    else:
        raise ValueError(
            f"Parameter data_loading.train_or_test in config should be 'train' or 'test', \
            got {config['data_loading']['train_or_test']}"
        )
    data_json = json.load(open(data_path))["data"]
    df = json_to_dataframe(data_json, config["data_loading"]["drop_impossible"])
    return df


def json_to_dataframe(data_json: dict, drop_impossible: bool):
    """Converts the SQuAD dataset from json to a pd.DataFrame.
    Works with versions 1.1 and 2.0 of the dataset.

    Args:
        data_json (dict): the SQuAD dataset as a json (dict)
        drop_impossible (bool): used only for version 2.0, if true :
        the unanswerable questions are dropped, if false : the unanswerable
        questions are merged with the answerable one

    Returns:
        pd.DataFrame:: DataFrame containing all the contexts, questions and
        answers. The columns are : 'answers', 'question', 'id', 'title',
        'context' and 'context_id'
    """
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


class SquadContexts(torch.utils.data.Dataset):
    """Dataset of the contexts of SQuAD, mainly used
    when computing the context embeddings.
    """

    def __init__(self, data_df):

        self.contexts = data_df["context"].unique()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx]

    def truncate_beginning(self, start_index: int):
        """Removes the beginning of the dataset, so that dataset[start index]
        is the new first element.

        Args:
            start_index (int): the index of the new first element
        """
        self.contexts = self.contexts[start_index:]


class SquadQuestions(torch.utils.data.Dataset):
    """Dataset of the questions of SQuAD, mainly used
    when computing the question embeddings.
    """

    def __init__(self, config, data_df):
        self.full_df = data_df

        self.random_state = config["seed"]
        self.sample_df = self.full_df

    def __len__(self):
        return len(self.sample_df.index)

    def __getitem__(self, idx):
        # we return both the "name" (= idx of question in the full_df) and the question
        # to be able to store the embedding at the right row of the result matrix
        return self.sample_df.iloc[idx].name, self.sample_df["question"].iloc[idx]

    def reduce_to_sample(
        self, frac: float, previous_embeddings: Union[torch.Tensor, None] = None
    ):
        """Reduces the size of the dataset to a sample of questions whose embedding have not yet
        been computed. The size of the sample is frac * size initial dataset.

        Args:
            frac (float): percentage of the initial dataset that will make the sample
            previous_embeddings (Union[torch.Tensor, None], optional): Tensor containing the question
            embeddings that were computed during previous runs and saved in a .pt file. Defaults to None.
        """
        if previous_embeddings is None:
            self.sample_df = self.full_df.sample(
                frac=frac, random_state=self.random_state
            )
        else:
            questions_idx_with_computed_score = list(
                previous_embeddings.sum(dim=1).nonzero().squeeze().numpy()
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
