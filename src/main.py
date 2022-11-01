import sys

import yaml

from src.data_loading import load_squad_to_df, SquadContexts
from src.models.model import load_model
from src.results_file import create_results_folder

from torch.utils.data import DataLoader


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

    dataset = SquadContexts(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    bert_model = BertEmbeddings(config)

    # TODO : complete evaluation of BERT


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
    else : 
        raise ValueError(f"Unknown model : {config["model"]["name"]}")

