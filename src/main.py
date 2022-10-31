import sys

import yaml

from src.data_loading import load_squad
from src.models.model import load_model
from src.results_file import create_results_folder


def main(config):
    create_results_folder(config)
    print("Loading dataset...")
    train_df = load_squad(config)
    # test_df = load_squad(config, test=True)
    print("Loading model...")
    model = load_model(config, train_df)
    print("Computing recall...")
    recall = model.compute_recall()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "examples/config.yaml"
    config = yaml.safe_load(open(config_path))

    main(config)
