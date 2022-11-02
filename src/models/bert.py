import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import os
import numpy as np
import pandas as pd
from src.data_loading import SquadContexts


class BertEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        pretrained_model_name = config["model_parameters"]["bert"][
            "pretrained_model_name"
        ]

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.pretrained_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # base bert output
        self.output_dim = self.bert.config.hidden_size
        self.config = config

    def forward(self, list_sentences):
        tokenized_sentences = self.tokenizer(list_sentences)
        # model_output dimensions : batch_size * nb tokens * bert_output_dim (=786)
        model_output = self.bert(**tokenized_sentences).last_hidden_state.detach()

        # to compute dot products between questions and paragraphs we need them all to have the same
        # dimension : we apply a mean over the dimension that has the size of nb tokens (~= nb of words)
        model_output = torch.mean(model_output, dim=1)
        return model_output

    def tokenizer(self, list_sentences):
        return self.pretrained_tokenizer(
            list_sentences, return_tensors="pt", padding=True, truncation=True
        )


def load_context_embeddings(bert_model, config):
    result_path = config["model_parameters"]["bert"]["context_embeddings_path"]

    if os.path.exists(result_path):
        result = torch.load(result_path)
        # to find where is the last computed embedding (in case program was stopped in the middle),
        # we suppose that if the sum of the composants of the embedding equals 0, it wasn't computed
        sum_of_embeddings_per_context = list(result.sum(dim=1).numpy())
        if 0.0 not in sum_of_embeddings_per_context:
            return result  # means the result matrix is already completely computed

        # If we need to continue from the middle :
        context_dataset = SquadContexts(config)
        start_index = sum_of_embeddings_per_context.index(0.0)

    else:
        context_dataset = SquadContexts(config)
        os.makedirs(os.path.join(*result_path.split("/")[:-1]), exist_ok=True)
        result = torch.zeros(len(context_dataset), bert_model.output_dim)
        start_index = 0

    context_dataset.truncate_beginning(start_index)
    batch_size = config["model_parameters"]["bert"]["batch_size"]
    dataloader = DataLoader(
        context_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch_idx : {batch_idx} / {len(context_dataset)//batch_size}", end="\r")
        embeddings = bert_model(batch)
        result[
            start_index
            + batch_idx * batch_size : start_index
            + (batch_idx + 1) * batch_size
        ] = embeddings
        torch.save(result, result_path)

    return result


def compute_question_embeddings(
    bert_model, question_dataset, config, context_embeddings
):
    result_path = config["model_parameters"]["bert"]["question_embeddings_path"]

    batch_size = config["model_parameters"]["bert"]["batch_size"]
    dataloader = DataLoader(
        question_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    result = torch.zeros(len(question_dataset.full_df.index), bert_model.output_dim)

    for batch_idx, batch in enumerate(dataloader):
        print(
            f"Batch_idx : {batch_idx} / {len(question_dataset)//batch_size}", end="\r"
        )
        questions_idx = batch[0]
        questions = batch[1]
        question_embeddings = bert_model(questions)
        result[questions_idx] = question_embeddings
        torch.save(result, result_path)


def compute_scores(config):
    question_embeddings = torch.load(
        config["model_parameters"]["bert"]["question_embeddings_path"]
    )
    context_embeddings = torch.load(
        config["model_parameters"]["bert"]["context_embeddings_path"]
    )
    scores_path = config["model_parameters"]["bert"]["similarity_score_path"]
    import time

    tic = time.time()
    print("TIC")
    scores = torch.tensordot(
        question_embeddings, torch.transpose(context_embeddings, 0, 1), dims=1
    )
    touc = time.time() - tic
    print("TOUC", touc)
    torch.save(scores, scores_path)
    tac = time.time() - tic
    print("TAC", tac)


def compute_metrics(config):
    bert_predictions_path = config["model_parameters"]["bert"]["prediction_df_path"]
    if not os.path.exists(bert_predictions_path):
        raise FileNotFoundError(
            f"Scores must be computed in {bert_predictions_path} before computing metrics.\
            We recommend relaunching the program once with 'display_saved_metrics_only = false' in the config file."
        )
    full_df = pd.read_csv(bert_predictions_path)
    df = full_df[~full_df["scores"].isna()]
    df["scores"] = df["scores"].apply(eval)
    df["scores"] = df["scores"].apply(np.array)

    df["predicted_context_id"] = df["scores"].apply(np.argmax)
    df["rank_of_true_context"] = df.apply(
        lambda x: list(np.argsort(-x["scores"])).index(x["context_id"]), axis=1
    )  # the minus sign allows to sort in descending order
    for i in [1, 5, 10]:
        print(
            f"True context in first {i}th contexts : {len(df[df['rank_of_true_context'] < i])} / {len(df.index)}   ({len(df[df['rank_of_true_context'] < i])/len(df.index) * 100}%)"
        )
    print("Mean rank of true context :", df["rank_of_true_context"].mean())


if __name__ == "__main__":
    import yaml

    config_path = "./../examples/config.yaml"
    config = yaml.safe_load(open(config_path))

    bert_model = BertModel(config)
