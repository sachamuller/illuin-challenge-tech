import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from src.data_loading import SquadContexts, SquadQuestions


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
        # model_output dimensions : batch_size * nb tokens * bert_output_dim (=768)
        model_output = self.bert(**tokenized_sentences).last_hidden_state.detach()

        # to compute dot products between questions and paragraphs we need them all to have the same
        # dimension : we apply a mean over the dimension that has the size of nb tokens (~= nb of words)
        model_output = torch.mean(model_output, dim=1)
        return model_output

    def tokenizer(self, list_sentences):
        return self.pretrained_tokenizer(
            list_sentences, return_tensors="pt", padding=True, truncation=True
        )


def load_context_embeddings(
    bert_model: BertEmbeddings, config: dict, data_df: pd.DataFrame
) -> torch.Tensor:
    """Returns the context embeddings.
    If an embedding file is already saved, it is loaded and if all the embeddings
    are already computed, the resulting tensor is simply returned.
    If no embedding file exists or it is incomplete (error during previous run),
    the missing embeddings will be computed and saved.

    Args:
        bert_model (BertEmbeddings): pretrained BERT model
        config (dict): config dictionnary coming from yaml file
        data_df (pd.DataFrame): DataFrame containing the SQuAD dataset

    Returns:
        torch.Tensor: tensor of size nb_contexts * bert_output_size (=768).
        The ith row if the embedding of the ith context.
    """
    print("Loading context embeddings...")

    result_path = config["model_parameters"]["bert"]["context_embeddings_path"]

    if os.path.exists(result_path):
        result = torch.load(result_path)
        # to find where is the last computed embedding (in case program was stopped in the middle),
        # we suppose that if the sum of the composants of the embedding equals 0, it wasn't computed
        sum_of_embeddings_per_context = list(result.sum(dim=1).numpy())
        if 0.0 not in sum_of_embeddings_per_context:
            return result  # means the result matrix is already completely computed

        # If we need to continue from the middle :
        context_dataset = SquadContexts(data_df)
        start_index = sum_of_embeddings_per_context.index(0.0)

    else:
        context_dataset = SquadContexts(data_df)
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
    bert_model: BertEmbeddings, config: dict, data_df: pd.DataFrame
) -> torch.Tensor:
    """Compute some question embeddings.
    If an embedding file is already saved, it is loaded. A sample of questions
    whose embeddings were not computed during a previous run is drawn and their
    embeddings are computed and added to the saved file.
    Same thing if no embeddings were saved, except we start from a blank tensor.

    Args:
        bert_model (BertEmbeddings): pretrained BERT model
        config (dict): config dictionnary coming from yaml file
        data_df (pd.DataFrame): DataFrame containing the SQuAD dataset

    Returns:
        torch.Tensor: tensor of size nb_questions * bert_output_size (=768)
        The ith row if the embedding of the ith question.
    """

    result_path = config["model_parameters"]["bert"]["question_embeddings_path"]

    question_dataset = SquadQuestions(config, data_df)
    if os.path.exists(result_path):
        result = torch.load(result_path)
        question_dataset.reduce_to_sample(
            config["model_parameters"]["bert"]["dataset_percentage"],
            result,
        )
    else:
        question_dataset.reduce_to_sample(
            config["model_parameters"]["bert"]["dataset_percentage"],
        )
        os.makedirs(os.path.join(*result_path.split("/")[:-1]), exist_ok=True)
        result = torch.zeros(len(question_dataset.full_df.index), bert_model.output_dim)

    batch_size = config["model_parameters"]["bert"]["batch_size"]
    dataloader = DataLoader(
        question_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for batch_idx, batch in enumerate(dataloader):
        print(
            f"Batch_idx : {batch_idx} / {len(question_dataset)//batch_size}", end="\r"
        )
        questions_idx = batch[0]
        questions = batch[1]
        question_embeddings = bert_model(questions)
        result[questions_idx] = question_embeddings
        torch.save(result, result_path)

    return result


def load_question_embeddings(
    bert_model: BertEmbeddings, config: dict, data_df: pd.DataFrame
) -> torch.Tensor:
    """Returns the question embeddings.
    If an embedding file is already saved, it is loaded and returned.
    If no embeddings were saved, the function to compute them is called.

    Args:
        bert_model (BertEmbeddings): pretrained BERT model
        config (dict): config dictionnary coming from yaml file
        data_df (pd.DataFrame): DataFrame containing the SQuAD dataset

    Returns:
        torch.Tensor: tensor of size nb_questions * bert_output_size (=768)
        The ith row if the embedding of the ith question.
    """
    print("Loading question embeddings...")
    if os.path.exists(config["model_parameters"]["bert"]["question_embeddings_path"]):
        return torch.load(
            config["model_parameters"]["bert"]["question_embeddings_path"]
        )
    else:
        return compute_question_embeddings(bert_model, config, data_df)


def compute_all_scores(
    bert_model: BertEmbeddings, config: dict, data_df: pd.DataFrame
) -> torch.Tensor:
    """Returns a matrix of size nb_questions * nb_contexts such as :
    scores[i][j] = dot product of the embeddings of the ith question and the jth context

    Args:
        bert_model (BertEmbeddings): pretrained BERT model
        config (dict): config dictionnary coming from yaml file
        data_df (pd.DataFrame): DataFrame containing the SQuAD dataset

    Returns:
        torch.Tensor: matrix containing the dot products of the embeddings
    """
    question_embeddings = load_question_embeddings(bert_model, config, data_df)
    context_embeddings = load_context_embeddings(bert_model, config, data_df)

    scores = compute_score(question_embeddings, context_embeddings)

    return scores


def compute_score(
    question_embeddings: torch.Tensor, context_embeddings: torch.Tensor
) -> torch.Tensor:
    """Returns the dot product of some question embeddings with
    some context embeddings

    Args:
        question_embeddings (torch.Tensor): tensor such as the ith row
        is the embedding of the ith question
        context_embeddings (torch.Tensor): tensor such as the ith row
        is the embedding of the ith context

    Returns:
        torch.Tensor: dot product of the two embeddings tensors
    """
    return torch.tensordot(
        question_embeddings, torch.transpose(context_embeddings, 0, 1), dims=1
    )


def compute_metrics(config: dict, df: pd.DataFrame, scores: torch.Tensor) -> None:
    """Compute for each question the rank of the true context, then prints
    the demanded metrics : how many questions give the true context in the top k
    predicted ones, and the mean rank of the true context.

    Args:
        config (dict): config dictionnary coming from yaml file
        df (pd.DataFrame): DataFrame containing the SQuAD dataset
        scores (torch.Tensor): tensor of size nb_questions * nb_contexts such as :
        scores[i][j] = dot product of the embeddings of the ith question and the jth context
    """

    # if the sum of the embedding is 0, we estimate it was not computed
    questions_idx_with_computed_score = list(
        scores.sum(dim=1).nonzero().squeeze().numpy()
    )

    scores = scores[questions_idx_with_computed_score]

    context_idx_array = np.array(
        df.loc[questions_idx_with_computed_score, "context_id"]
    )
    # We create a matrix of the same size as score (=nb_questions*nb_contexts) such as :
    #  matrix[i][j] = context_idx of question i
    repeated_context_idx_array = np.repeat(
        [context_idx_array], scores.shape[1], axis=0
    ).transpose()

    # We argsort the matrix of scores for each row meaning :
    # row[k] is the index of the kth biggest score
    argsort_score = (-scores).argsort(axis=1).numpy()

    df.loc[questions_idx_with_computed_score, "rank_of_true_context"] = np.where(
        argsort_score == repeated_context_idx_array
    )[1]

    total = len(questions_idx_with_computed_score)
    for i in config["metrics"]["list_top_k"]:
        print(
            f"True context in first {i}th contexts : {len(df[df['rank_of_true_context'] < i])} \
/ {total}   ({round(len(df[df['rank_of_true_context'] < i])/total * 100, 2)}%)"
        )
    print("Mean rank of true context :", df["rank_of_true_context"].mean())


def predict_context_for_one_question(
    question: str,
    context_embeddings: torch.Tensor,
    bert_model: BertEmbeddings,
    data_df: pd.DataFrame,
) -> str:
    """Given a question, returns the context most likely to contain the answer.

    Args:
        question (str): a question in natural language
        context_embeddings (torch.Tensor): tensor such as the ith row is the
        embedding of the ith context.
        bert_model (BertEmbeddings): pre-trained BERT model
        data_df (pd.DataFrame): DataFrame containing the SQuAD dataset

    Returns:
        str: the context most likely to contain the answer
    """
    question_embeddings = bert_model([question])
    score = compute_score(question_embeddings, context_embeddings)
    predicted_context_id = score.argmax().item()
    return data_df[data_df["context_id"] == predicted_context_id]["context"].iloc[0]


if __name__ == "__main__":
    import yaml

    config_path = "./../examples/config.yaml"
    config = yaml.safe_load(open(config_path))

    bert_model = BertModel(config)
