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
        self.top_n = config["model_parameters"]["bm25"]["top_n"]
        self.save_folder = config["results"]["result_folder_name"]

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
        log: bool = True,
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
            lambda x: self.model.get_top_n(x, self.data["context"].unique(), n=top_n)
        )

        sampled_data["score"] = sampled_data.apply(
            lambda row: row["context"] in row["prediction"], axis=1
        )
        recall = sum(sampled_data["score"]) / len(sampled_data.index)
        tac = time.time()
        if log or verbose:
            log_string = ""
            log_string += f"Size of sample : {len(sampled_data.index)}\n"
            log_string += f"Correct guesses : {sum(sampled_data['score'])}\n"
            for i in range(top_n):
                ith_correct = sum(
                    sampled_data.apply(
                        lambda row: row["context"] == row["prediction"][i], axis=1
                    )
                )
                log_string += f"{i+1}th place : {ith_correct} ({round(ith_correct/len(sampled_data.index)*100, 2)}%)\n"
            log_string += f"Recall : {recall}\n"
            log_string += f"Computation time : {tac - tic}\n"
        if verbose:
            print(log_string)
        if log:
            with open(os.path.join(self.save_folder, "results.txt"), "w+") as f:
                f.write(log_string)
        return recall
