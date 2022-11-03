def adapt_path_names(config: dict) -> None:
    """Changes some file names in the config
    This function replaces '{train_or_test}' by 'train' or 'test', according
    to whether the program was launched on the train or the test dataset.
    This is necessary not to overwrite the embeddings of one dataset with the embeddings
    of the other dataset.

    Args:
        config (dict): config dictionnary coming from yaml file
    """
    train_or_test = config["data_loading"]["train_or_test"]
    config["model_parameters"]["bert"]["context_embeddings_path"] = config[
        "model_parameters"
    ]["bert"]["context_embeddings_path"].replace("{train_or_test}", train_or_test)
    config["model_parameters"]["bert"]["question_embeddings_path"] = config[
        "model_parameters"
    ]["bert"]["question_embeddings_path"].replace("{train_or_test}", train_or_test)
