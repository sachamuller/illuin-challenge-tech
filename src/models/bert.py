import torch
from transformers import BertTokenizer, BertModel


class BertEmbeddings(torch.nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        pretrained_model_name = config["model_parameters"]["bert"][
            "pretrained_model_name"
        ]

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.pretrained_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # base bert output
        self.output_dim = self.bert.config.hidden_size
        self.config = config

    def forward(self, list_sentences):
        tokenized_sentences = self.tokenizer(list_sentences)
        # model_output dimensions : batch_size * nb tokens * bert_output_dim (=786)
        model_output = self.bert(**tokenized_sentences).last_hidden_state.detach()

        # to compute dot products between questions and paragraphs we need them all to have the same
        # dimension : we apply a mean over the dimension that has the size of nb tokens (~= nb of words)
        model_output = torch.mean(model_output, dim=1)
        return model_output

    def tokenizer(self, list_sentences):
        return self.pretrained_tokenizer(
            list_sentences, return_tensors="pt", padding=True, truncation=True
        )


if __name__ == "__main__":
    config_path = "./../examples/config.yaml"
    config = yaml.safe_load(open(config_path))

    bert_model = BertModel(config)
