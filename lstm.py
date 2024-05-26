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
    return [torch.tensor([vocabulary["<bos>"]] + [vocabulary[token] for token in document] + [vocabulary["<eos>"]],
                         dtype=torch.long) for document in dt]


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


def get_y_pred(tokens):
    y_pred = []
    for j in tqdm(range(len(tokens))):
        tok = lstm_model(tokens[j])
        idx = torch.argmax(tok, 1)
        lst = list(idx.numpy())
        y_pred.append(lst)
    return y_pred


def y_pred_to_labels(y_pred):
    labels = []
    for record in y_pred:
        rec_labels = []
        for num in record:
            label = None
            for a, b in label_vocab.items():
                if num == b:
                    label = a
            rec_labels.append(label)
        labels.append(rec_labels)
        del rec_labels[0]
        del rec_labels[-1]
    return labels


train_data = pd.read_csv('train/train.tsv', sep='\t', names=["labels", "texts"], header=None)
validation_data = pd.read_csv('dev-0/in.tsv', sep='\t', names=["texts"], header=None)
validation_data_labels = pd.read_csv('dev-0/expected.tsv', sep='\t', names=["labels"], header=None)
test_data = pd.read_csv('test-A/in.tsv', sep='\t', names=["texts"], header=None)

train_words_split = []
for text in train_data["texts"]:
    row = []
    for word in text.split(" "):
        row.append(word)
    train_words_split.append(row)

validation_words_split = []
for text in validation_data["texts"]:
    row = []
    for word in text.split(" "):
        row.append(word)
    validation_words_split.append(row)

test_words_split = []
for text in test_data["texts"]:
    row = []
    for word in text.split(" "):
        row.append(word)
    test_words_split.append(row)


vocabulary = build_vocab(train_words_split)
vocabulary.set_default_index(vocabulary["<unk>"])
itos = vocabulary.get_itos()


label_vocab = {}
count = 0
for text in train_data["labels"]:
    for label in text.split(" "):
        if label not in label_vocab:
            label_vocab[label] = count
            count += 1


labels_train_data = []
for text in train_data["labels"]:
    row = []
    for label in text.split(" "):
        row.append(label_vocab[label])
    labels_train_data.append(row)

labels_validation_data = []
for text in validation_data_labels["labels"]:
    row = []
    for label in text.split(" "):
        row.append(label_vocab[label])
    labels_validation_data.append(row)


train_tokens_ids = data_process(train_words_split)
validation_tokens_ids = data_process(validation_words_split)
test_tokens_ids = data_process(test_words_split)

train_labels = labels_process(labels_train_data)
validation_labels = labels_process(labels_validation_data)


num_tags = len(label_vocab.keys())


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.emb = torch.nn.Embedding(len(vocabulary.get_itos()), 100)
        self.rec = torch.nn.LSTM(100, 256, 1, batch_first=True)
        self.fc1 = torch.nn.Linear(256, num_tags)

    def forward(self, x):
        emb = torch.relu(self.emb(x))
        lstm_output, (h_n, c_n) = self.rec(emb)
        out_weights = self.fc1(lstm_output)
        return out_weights


lstm_model = LSTM()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters())
NUM_EPOCHS = 10

for i in range(NUM_EPOCHS):
    lstm_model.train()
    for k in tqdm(range(len(train_labels))):
        batch_tokens = train_tokens_ids[k].unsqueeze(0)
        tags = train_labels[k].unsqueeze(1)

        predicted_tags = lstm_model(batch_tokens)

        optimizer.zero_grad()
        loss = criterion(predicted_tags.squeeze(0), tags.squeeze(1))

        loss.backward()
        optimizer.step()

    lstm_model.eval()
    print(eval_model(validation_tokens_ids, validation_labels, lstm_model))


print(eval_model(validation_tokens_ids, validation_labels, lstm_model))

# (0.9501402247360199, 0.9508740204942736, 0.9505069809917434)

y_pred_validation = get_y_pred(validation_tokens_ids)
y_pred_validation_labels = y_pred_to_labels(y_pred_validation)

output = pd.DataFrame({'predicted_label': [' '.join(map(str, row)) for row in y_pred_validation_labels]})
output.to_csv("dev-0/out.tsv", sep='\t', index=False, header=False)

y_pred_test = get_y_pred(test_tokens_ids)
y_pred_test_labels = y_pred_to_labels(y_pred_test)

output2 = pd.DataFrame({'predicted_label': [' '.join(map(str, row)) for row in y_pred_test_labels]})
output2.to_csv("test-A/out.tsv", sep='\t', index=False, header=False)
