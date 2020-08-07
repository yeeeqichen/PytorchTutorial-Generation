import ReadCorpus
import random
import torch
from Config import config


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


class Loader:
    def __init__(self):
        self.all_categories = ReadCorpus.all_categories
        self.n_categories = ReadCorpus.n_categories
        self.names = ReadCorpus.category_lines
        self.n_letters = ReadCorpus.n_letters
        self.all_letters = ReadCorpus.all_letters

    def get_training_pair(self):
        category = random_choice(self.all_categories)
        name = random_choice(self.names[category])
        return category, name

    def to_tensor(self, category, name):
        category_tensor = torch.zeros(1, self.n_categories)
        category_tensor[0][self.all_categories.index(category)] = 1
        name_tensor = torch.zeros(len(name), 1, self.n_letters)
        target_idx = [self.all_letters.find(l) for l in name[1:]]
        target_idx.append(self.n_letters - 1)
        target_tensor = torch.LongTensor(target_idx)
        for i in range(len(name)):
            letter = name[i]
            name_tensor[i][0][self.all_letters.find(letter)] = 1
        category_tensor = category_tensor.to(config.device)
        name_tensor = name_tensor.to(config.device)
        target_tensor = target_tensor.to(config.device)
        return category_tensor, name_tensor, target_tensor

    def get_training_data(self):
        category, name = self.get_training_pair()
        return self.to_tensor(category, name)


loader = Loader()
# print(loader.get_training_data())

