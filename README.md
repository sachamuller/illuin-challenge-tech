# Illuin Challenge Tech
---

## :information_desk_person: Project description

The goal of this project is to find the context which will more likely contain the answer to a question, among all the contexts of a dataset. It was conducted as part of the recruitment process of Illuin Technology. 

Two solutions were implemented with two different models : one solution using BM25 and one solution using BERT.

For each model, the program can be launched in evaluation mode or in predict mode :
-  In evaluation mode, the program computes the performance of the model over all (or part of) the questions of the dataset.
- In predict mode, the user can enter questions in the terminal and the model will provide the predicted context. 


---

## :construction: Installation
1. Clone the current repository
2. We recommand creating a new virtual environment, then installing the required packages with : 
```
pip install -r requirements.txt
```
3. Download the [SQuAD Dataset](https://deepai.org/dataset/squad). This program technically works with version 1.1 and 2.0 of the dataset, but we recommend using version 1.1 as version 2.0 only adds unanswerable questions, which are not used in the program.
4. Unzip the downloaded archive and place the `squadX.X` folder in the `data` folder.
5. Everything is ready, the program can now be launched !



---
## Usage

To run the program in predict mode use :
```bash
python -m src.main --predict
```
To run the program in evaluation mode use :
```bash
python -m src.main --evaluate
```

The other parameters, including the model used, can be changed in the config file, which is by default `examples/config.yaml`. You can also launch the program using another config file with the `--config_file` argument : 
```bash
python -m src.main --config_file path_of_your_file/config.yaml
```

To chose which model to use, you need to modify the `model.name` parameter in the config file : 
```yaml
model :
  name: bm25
```

The other parameters are detailed below. 

---
## Configuration file 

- `data_loading`:
  - `train_or_test`: 'train' or 'test', whether to use the train dataset (`train-vX.X.json`) or the test dataset (`dev-vX.X.json`)
  - `train_path`: string, path of the train dataset
  - `test_path`: string, path of the test dataset
  - `drop_impossible`: boolean, only used for the version 2.0 of the dataset : whether to drop the unanswerable questions, if false, the unanswerable questions are merged with the answerable one.

- `model` :
  - `name`: 'bm25' or 'bert', the model usef

- `model_parameters` :
  - `bm25` :
    - `dataset_percentage`: float, the percentage of questions used to compute the metrics in evaluation mode
  - `bert` :
    - `pretrained_model_name`: string, value used when calling `BertModel.from_pretrained(pretrained_model_name)`
    - `context_embeddings_path`: string, path of the `.pt` file in which the embeddings of all the contexts will be saved
    - `question_embeddings_path`: string, same for question embeddings
    - `batch_size`: int, batch size used when computing the embeddings (context and question), the results are saved at the end of each batch
    - `dataset_percentage`: float, the percentage of questions used to compute the metrics in evaluation mode
    - `compute_more_question_embeddings`: boolean, if false : the first time the program is launched it will compute the embeddings of a percentage of the questions (given by `dataset_percentage`), then the next time it is launched, it will only load the embeddings that were computed the first time. If true : each time the program is launched, it will load the previously computed embeddings and recompute new embeddings (quantity given by `dataset_percentage`) (until eventually all the question embeddings are computed and saved)

`metrics` : 
  `list_top_k`: list\[int\], during evaluation, will print the number of questions for which the true context is in the top k predicted contexts

`seed` : int, used to get reproducibility, the random aspect of the program being the selection of the questions that will have their embeddings computed when `dataset_percentage` $< 1.0$.


