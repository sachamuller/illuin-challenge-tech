import time

from rank_bm25 import BM25Okapi
from typing import Union


class bm25_model:
    def __init__(self, config, data) -> None:
        data["tokenized_question"] = data["question"].apply(lambda x: x.split(" "))
        self.data = data
        self.tokenized_corpus = [doc.split(" ") for doc in data["context"]]
        self.model = BM25Okapi(self.tokenized_corpus)
        self.random_state = config["seed"]
        self.sample_frac = config["model_parameters"]["bm25"]["dataset_percentage"]
        self.top_n = config["model_parameters"]["bm25"]["top_n"]

    def predict(self, question: str):
        tokenized_question = question.split(" ")
        return self.model.get_top_n(
            tokenized_question, list(self.data["context"]), n=1
        )[0]

    def compute_recall(
        self,
        sample_frac: Union[float, None] = None,
        top_n: Union[int, None] = None,
        verbose: bool = True,
    ):
        if sample_frac is None:
            sample_frac = self.sample_frac
        if top_n is None:
            top_n = self.top_n

        tic = time.time()
        sampled_data = self.data.sample(
            frac=sample_frac, random_state=self.random_state
        )
        sampled_data["prediction"] = sampled_data["tokenized_question"].apply(
            lambda x: self.model.get_top_n(x, list(self.data["context"]), n=top_n)
        )

        sampled_data["score"] = sampled_data.apply(
            lambda row: row["context"] in row["prediction"], axis=1
        )
        recall = sum(sampled_data["score"]) / len(sampled_data.index)
        tac = time.time()
        if verbose:
            print("Size of sample :", len(sampled_data.index))
            print("Correct guesses :", sum(sampled_data["score"]))
            print("Recall :", recall)
            print("Computation time :", tac - tic)
        return recall
