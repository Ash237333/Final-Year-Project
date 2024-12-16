import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

VOCAB_SIZE = 30000
BATCH_SIZE = 8
local_dataset_path = "wmt/wmt14"

def train_tokenizer():
    ds = load_dataset(local_dataset_path, split="train", streaming="true")
    def dataset_iterator(dataset):
        for i in dataset:
            yield i["translation"]["de"]
            yield i["translation"]["en"]

    BPE_tokenizer = Tokenizer(BPE())
    BPE_trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True, special_tokens=["[PAD]", "[UNK]"])
    BPE_tokenizer.train_from_iterator(dataset_iterator(ds),BPE_trainer, length = 300000*2)
    BPE_tokenizer.save("BPE_Tokenizer.json")


def load_dataset_and_preprocess():
    train_dataset = load_dataset(local_dataset_path, "de-en", split="train")
    test_dataset = load_dataset(local_dataset_path,"de-en", split="test")

    test_dataset = tokenize_dataset(test_dataset)
    train_dataset = tokenize_dataset(train_dataset)

    return train_dataset, test_dataset


def tokenize_dataset(dataset):
    BPE_tokenizer = PreTrainedTokenizerFast(tokenizer_file="BPE_Tokenizer.json")

    def tokenize_func(example):
        gtext_tokenized = BPE_tokenizer.encode(example["translation"]["de"])
        etext_tokenized = BPE_tokenizer.encode(example["translation"]["en"])
        return {
            "input_ids": gtext_tokenized,
            "labels": etext_tokenized
        }

    tokenized_dataset = dataset.map(tokenize_func, remove_columns=["translation"])
    return tokenized_dataset


def create_dataloader():
    train, test = load_dataset_and_preprocess()
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader


def collate_fn(batch):
    input_ids, attention_mask, labels = [], [], []
    for i in batch:
        input_ids.append(torch.tensor(i["input_ids"]))
        labels.append(torch.tensor(i["labels"]))

    padded_input_ids = pad_sequence(input_ids, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)

    return padded_input_ids, padded_labels

train_dataset = load_dataset(local_dataset_path, "de-en", split="train")
print(len(train_dataset))