import ReadCorpus
import torch


class Config:
    def __init__(self):
        self.hidden_size = 128
        self.vocab_size = ReadCorpus.n_letters
        self.categories_size = ReadCorpus.n_categories
        self.learning_rate = 0.0001
        self.EPOCH = 5
        self.iter_per_epoch = 10000
        self.batch_size = 50
        self.dropout = 0.1
        self.output_size = self.vocab_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.max_length = 15


config = Config()
