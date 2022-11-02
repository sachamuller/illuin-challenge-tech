import os
import time
from typing import Union

from rank_bm25 import BM25Okapi


class bm25_model:
    def __init__(self, config, data) -> None:
        data["tokenized_question"] = data["question"].apply(lambda x: x.split(" "))
        self.data = data
        self.tokenized_corpus = [doc.split(" ") for doc in data["context"].unique()]
        self.model = BM25Okapi(self.tokenized_corpus)
        self.random_state = config["seed"]
        self.sample_frac = config["model_parameters"]["bm25"]["dataset_percentage"]
        self.list_top_k = config["metrics"]["list_top_k"]

    def predict(self, question: str):
        tokenized_question = question.split(" ")
        return self.model.get_top_n(
            tokenized_question, list(self.data["context"]), n=1
        )[0]

    def compute_recall(
        self,
        sample_frac: Union[float, None] = None,
    ):
        if sample_frac is None:
            sample_frac = self.sample_frac

        sampled_data = self.data.sample(
            frac=sample_frac, random_state=self.random_state
        )
        sampled_data["prediction"] = sampled_data["tokenized_question"].apply(
            lambda x: self.model.get_top_n(
                x, self.data["context"].unique(), n=len(self.data.context.unique())
            )
        )

        sampled_data["rank_of_true_context"] = sampled_data.apply(
            lambda x: x["prediction"].index(x["context"]), axis=1
        )

        total = len(sampled_data.index)
        for i in self.list_top_k:
            print(
                f"True context in first {i}th contexts : {len(sampled_data[sampled_data['rank_of_true_context'] < i])} \
/ {total}   ({round(len(sampled_data[sampled_data['rank_of_true_context'] < i])/total * 100, 2)}%)"
            )
        print(
            "Mean rank of true context :", sampled_data["rank_of_true_context"].mean()
        )
