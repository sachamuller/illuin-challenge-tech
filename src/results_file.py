import json
import os
from datetime import datetime


def create_results_folder(config):
    parent_folder = config["results"]["parent_folder_path"]
    if config["results"]["result_folder_name"] == "datetime":
        result_folder_name = config["results"]["result_folder_name"] = config["model"][
            "name"
        ] + datetime.now().strftime("_%d-%m-%Y_%H:%M:%S")
    else:
        result_folder_name = config["results"]["result_folder_name"]
    config["results"]["result_folder_name"] = os.path.join(
        parent_folder, result_folder_name
    )
    os.makedirs(
        config["results"]["result_folder_name"], exist_ok=config["results"]["exist_ok"]
    )
    json.dump(
        config,
        open(
            os.path.join(config["results"]["result_folder_name"], "config.json"), "w+"
        ),
        indent=4,
    )
