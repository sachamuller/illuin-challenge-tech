import sys

import yaml

from src.data_loading import load_squad_to_df, SquadContexts, SquadQuestions
from src.results_file import create_results_folder
from src.models.bm25 import bm25_model
from src.models.bert import BertEmbeddings, load_context_embeddings, compute_scores


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
    create_results_folder(config)
    print("Loading context dataset...")
    dataset = SquadContexts(config)
    print("Loading model...")
    bert_model = BertEmbeddings(config)
    print("Loading context embeddings...")
    context_embeddings = load_context_embeddings(bert_model, dataset, config)
    print(context_embeddings)

    print("Loading questions dataset...")
    questions = SquadQuestions(config)
    questions.reduce_to_sample(config["model_parameters"]["bert"]["dataset_percentage"])

    print("Compute scores")
    compute_scores(bert_model, questions, config, context_embeddings)


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
