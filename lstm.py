from collections import Counter
from tqdm import tqdm
import pandas as pd
from torchtext.vocab import vocab
import torch


def build_vocab(dataset):
    counter = Counter()
    for document in dataset:
        counter.update(document)
    return vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def data_process(dt):
    return [torch.tensor([v["<bos>"]] + [v[token] for token in document] + [v["<eos>"]], dtype=torch.long)
            for document in dt]


def labels_process(dt):
    return [torch.tensor([0] + document + [0], dtype=torch.long) for document in dt]


def get_scores(y_true, y_pred):
    acc_score = 0
    tp = 0
    fp = 0
    selected_items = 0
    relevant_items = 0

    for p, t in zip(y_pred, y_true):
        if p == t:
            acc_score += 1

        if p > 0 and p == t:
            tp += 1

        if p > 0:
            selected_items += 1

        if t > 0:
            relevant_items += 1

    if selected_items == 0:
        precision = 1.0
    else:
        precision = tp / selected_items

    if relevant_items == 0:
        recall = 1.0
    else:
        recall = tp / relevant_items

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def eval_model(dataset_tokens, dataset_labels, model):
    Y_true = []
    Y_pred = []
    for i in tqdm(range(len(dataset_labels))):
        batch_tokens = dataset_tokens[i].unsqueeze(0)
        tags = list(dataset_labels[i].numpy())
        Y_true += tags

        Y_batch_pred_weights = model(batch_tokens).squeeze(0)
        Y_batch_pred = torch.argmax(Y_batch_pred_weights, 1)
        Y_pred += list(Y_batch_pred.numpy())

    return get_scores(Y_true, Y_pred)


train_data = pd.read_csv('train/train.tsv', sep='\t', names=["labels", "texts"], header=None)
validation_data = pd.read_csv('dev-0/in.tsv', sep='\t', names=["texts"], header=None)
test_data = pd.read_csv('test-A/in.tsv', sep='\t', names=["texts"], header=None)

train_words = []
for tekst in train_data["texts"]:
    pom = []
    for slowo in tekst.split(" "):
        pom.append(slowo.lower())
    train_words.append(pom)

validation_words = []
for tekst in validation_data["texts"]:
    pom = []
    for slowo in tekst.split(" "):
        pom.append(slowo.lower())
    validation_words.append(pom)

test_words = []
for tekst in test_data["texts"]:
    pom = []
    for slowo in tekst.split(" "):
        pom.append(slowo.lower())
    test_words.append(pom)


v = build_vocab(train_words)
v.set_default_index(v["<unk>"])
itos = v.get_itos()
print(len(itos))
print(itos)


etykieta_na_kod = {}
licznik = 0
for tekst in train_data["labels"]:
    for etykieta in tekst.split(" "):
        if etykieta not in etykieta_na_kod:
            etykieta_na_kod[etykieta] = licznik
            licznik += 1
print(etykieta_na_kod)


kody_etykiet_train = []
for tekst in train_data["labels"]:
    pom = []
    for etykieta in tekst.split(" "):
        pom.append(etykieta_na_kod[etykieta])
    kody_etykiet_train.append(pom)
print(kody_etykiet_train[0])

# odczytaj etykiety dev-0
labels_dev0 = pd.read_csv('dev-0/expected.tsv', sep='\t')
labels_dev0.columns = ["y"]
print(labels_dev0["y"][0])

# podziel etykiety
kody_etykiet_dev0 = []
for tekst in labels_dev0["y"]:
    pom = []
    for etykieta in tekst.split(" "):
        pom.append(etykieta_na_kod[etykieta])
    kody_etykiet_dev0.append(pom)
print(kody_etykiet_dev0[0])


train_tokens_ids = data_process(train_words)
test_dev0_tokens_ids = data_process(validation_words)
test_A_tokens_ids = data_process(test_words)

train_labels = labels_process(kody_etykiet_train)
test_dev0_labels = labels_process(kody_etykiet_dev0)


print(len(train_tokens_ids), len(train_tokens_ids[0]))
print(len(test_dev0_tokens_ids), len(test_dev0_tokens_ids[0]))
print(len(test_A_tokens_ids), len(test_A_tokens_ids[0]))

print(train_tokens_ids[0])
print(test_dev0_tokens_ids[0])
print(test_A_tokens_ids[0])

print(len(train_labels), len(train_labels[0]))
print(len(test_dev0_labels), len(test_dev0_labels[0]))

print(train_labels[0])
print(test_dev0_labels[0])


num_tags = len(etykieta_na_kod.keys())


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.emb = torch.nn.Embedding(len(v.get_itos()), 100)
        self.rec = torch.nn.LSTM(100, 256, 1, batch_first=True)
        self.fc1 = torch.nn.Linear(256, num_tags)

    def forward(self, x):
        emb = torch.relu(self.emb(x))
        lstm_output, (h_n, c_n) = self.rec(emb)
        out_weights = self.fc1(lstm_output)
        return out_weights


lstm = LSTM()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters())
NUM_EPOCHS = 5
for i in range(NUM_EPOCHS):
    lstm.train()
    for i in tqdm(range(len(train_labels))):
        batch_tokens = train_tokens_ids[i].unsqueeze(0)
        tags = train_labels[i].unsqueeze(1)

        predicted_tags = lstm(batch_tokens)

        optimizer.zero_grad()
        loss = criterion(predicted_tags.squeeze(0), tags.squeeze(1))

        loss.backward()
        optimizer.step()

    lstm.eval()


scores_dev_0 = eval_model(test_dev0_tokens_ids, test_dev0_labels, lstm)

print(scores_dev_0)

# output = pd.DataFrame({'predicted_label': [' '.join(map(str, row)) for row in Y_pred_dev_0]})
# output.to_csv("dev-0/out.tsv", sep='\t', index=False, header=False)
#
# output2 = pd.DataFrame({'predicted_label': [' '.join(map(str, row)) for row in Y_pred_testA]})
# output2.to_csv("test-A/out.tsv", sep='\t', index=False, header=False)

