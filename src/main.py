import sys
import yaml
import argparse

from src.data_loading import load_squad_to_df
from src.results_file import adapt_path_names
from src.models.bm25 import bm25_model
from src.models.bert import (
    BertEmbeddings,
    compute_question_embeddings,
    compute_all_scores,
    compute_metrics,
    predict_context_for_one_question,
    load_context_embeddings,
)


def main_bm25(config):
    print("Loading dataset...")
    train_df = load_squad_to_df(config)
    # test_df = load_squad_to_df(config, test=True)
    print("Loading model...")
    model = bm25_model(config, train_df)
    print("Computing recall...")
    model.compute_recall()


def evaluate_bert(config):
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


def bert_predict(config):
    adapt_path_names(config)
    print("Loading model...")
    bert_model = BertEmbeddings(config)
    print("Loading dataset...")
    data_df = load_squad_to_df(config)

    context_embeddings = load_context_embeddings(bert_model, config, data_df)

    question = input("Question : ")
    while len(question) > 0:
        context = predict_context_for_one_question(
            question, context_embeddings, bert_model, config, data_df
        )
        print(context)
        question = input("\nQuestion : ")


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="examples/config.yaml")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--evaluate", dest="predict", action="store_false")
parser.set_defaults(predict=True)
if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path))

    if config["model"]["name"] == "bm25":
        main_bm25(config)
    elif config["model"]["name"] == "bert":
        if args.predict:
            bert_predict(config)
        else:
            evaluate_bert(config)
    else:
        raise ValueError(f"Unknown model : {config['model']['name']}")
