def adapt_path_names(config):
    train_or_test = config["data_loading"]["train_or_test"]
    config["model_parameters"]["bert"]["context_embeddings_path"] = config[
        "model_parameters"
    ]["bert"]["context_embeddings_path"].replace("{train_or_test}", train_or_test)
    config["model_parameters"]["bert"]["question_embeddings_path"] = config[
        "model_parameters"
    ]["bert"]["question_embeddings_path"].replace("{train_or_test}", train_or_test)
