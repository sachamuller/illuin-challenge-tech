import sys

import yaml

from src.data_loading import load_squad
from src.models.model import load_model


def main(config):
    train_df = load_squad(config)
    # test_df = load_squad(config, test=True)
    model = load_model(config, train_df)
    recall = model.compute_recall(0.01)
    print("recall :", recall)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "examples/config.yaml"
    config = yaml.safe_load(open(config_path))

    main(config)
