import sys
import os
import yaml

from src.data_loading import load_squad_to_df, SquadContexts, SquadQuestions
from src.results_file import create_results_folder, adapt_path_names
from src.models.bm25 import bm25_model
from src.models.bert import (
    BertEmbeddings,
    load_context_embeddings,
    compute_question_embeddings,
    compute_scores,
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
    # print("Loading model...")
    # bert_model = BertEmbeddings(config)
    # print("Loading context embeddings...")
    # context_embeddings = load_context_embeddings(bert_model, config)

    # print("Loading context embeddings...")
    # context_embeddings = load_context_embeddings(bert_model, config)

    # bert_predictions_path = config["model_parameters"]["bert"]["prediction_df_path"]
    # if config["model_parameters"]["bert"][
    #     "always_compute_questions_embeddings"
    # ] or not os.path.exists(bert_predictions_path):
    #     print("Loading question embeddings...")
    #     questions = SquadQuestions(config)
    #     questions.reduce_to_sample(
    #         config["model_parameters"]["bert"]["dataset_percentage"],
    #         config["model_parameters"]["bert"]["new_samples_only"],
    #     )
    #     compute_question_embeddings(bert_model, questions, config, context_embeddings)

    scores = compute_scores(config)
    print("Metrics :")
    compute_metrics(config, scores)


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
