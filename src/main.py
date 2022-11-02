import sys
import os
import yaml

from src.data_loading import load_squad_to_df
from src.results_file import create_results_folder, adapt_path_names
from src.models.bm25 import bm25_model
from src.models.bert import (
    BertEmbeddings,
    compute_question_embeddings,
    compute_all_scores,
    compute_metrics,
)


def main_bm25(config):
    create_results_folder(config)
    print("Loading dataset...")
    train_df = load_squad_to_df(config)
    # test_df = load_squad_to_df(config, test=True)
    print("Loading model...")
    model = bm25_model(config, train_df)
    print("Computing recall...")
    recall = model.compute_recall()


def main_bert(config):
    adapt_path_names(config)
    print("Loading model...")
    bert_model = BertEmbeddings(config)
    print("Loading dataset...")
    data_df = load_squad_to_df(config)

    if config["model_parameters"]["bert"]["compute_more_question_embeddings"]:
        compute_question_embeddings(bert_model, config, data_df)

    scores = compute_all_scores(bert_model, config, data_df)
    print("Metrics :")
    compute_metrics(config, data_df, scores)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "examples/config.yaml"
    config = yaml.safe_load(open(config_path))

    if config["model"]["name"] == "bm25":
        main_bm25(config)
    elif config["model"]["name"] == "bert":
        main_bert(config)
    else:
        raise ValueError(f"Unknown model : {config['model']['name']}")
