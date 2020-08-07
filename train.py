from Config import config
from Model import model
from DataLoader import loader
import torch


def train():
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(lr=config.learning_rate, params=model.parameters())
    # torch.autograd.set_detect_anomaly(True)
    for i in range(config.EPOCH):
        for j in range(config.iter_per_epoch):
            hidden = model.init_hidden()
            optimizer.zero_grad()
            loss = 0
            category_tensor, name_tensor, target_tensor = loader.get_training_data()
            target_tensor.unsqueeze_(-1)
            for k in range(name_tensor.size(0)):
                output, hidden = model(category_tensor, name_tensor[k], hidden)
                # print(output.shape, target_tensor[k].shape)
                loss += loss_fn(output, target_tensor[k])
            loss.backward(retain_graph=True)
            optimizer.step()
            if j % 1000 == 0:
                print(loss)


def test(category, begin):
    def _get_letter_tensor(letter):
        letter_tensor = torch.zeros(1, loader.n_letters)
        letter_tensor[0][loader.all_letters.find(letter)] = 1
        return letter_tensor.to(config.device)
    def _get_category_tensor(category):
        category_tensor = torch.zeros(1, loader.n_categories)
        category_tensor[0][loader.all_categories.index(category)] = 1
        return category_tensor.to(config.device)
    result = begin
    hidden = model.init_hidden()
    input_tensor = _get_letter_tensor(begin)
    category_tensor = _get_category_tensor(category)
    for i in range(config.max_length):
        output, hidden = model(category_tensor, input_tensor, hidden)
        topv, topi = output.topk(1)
        topi = topi[0][0]
        # print(topi)
        if topi == loader.n_letters - 1:
            break
        result += loader.all_letters[topi]
        input_tensor = _get_letter_tensor(result[-1])
    return result


train()
for begin in "ABC":
    print(test('Arabic', begin))
    print(test('Chinese', begin))
    print(test('German', begin))
