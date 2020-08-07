import torch
from Config import config


class Rnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.i2o = torch.nn.Linear(config.vocab_size + config.hidden_size + config.categories_size, config.output_size)
        self.i2h = torch.nn.Linear(config.vocab_size + config.hidden_size + config.categories_size, config.hidden_size)
        self.o2o = torch.nn.Linear(config.output_size + config.hidden_size, config.output_size)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        # print(category.shape, input.shape, hidden.shape)
        combined = torch.cat((category, input, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), dim=1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    @staticmethod
    def init_hidden():
        return torch.zeros(1, config.hidden_size).to(config.device)


model = Rnn()
model.to(config.device)
print(model)
