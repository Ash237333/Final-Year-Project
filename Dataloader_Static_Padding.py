from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 16
local_dataset_path = "wmt/wmt14"
MAX_LENGTH = 512
BPE_tokenizer = PreTrainedTokenizerFast(tokenizer_file="BPE_Tokenizer.json")
BPE_tokenizer.pad_token = "[PAD]"


def create_dataloader():
    test_dataset = load_dataset(local_dataset_path,"de-en", split="test")
    test_dataset = tokenize_dataset(test_dataset)
    test_dataset = remove_long_data(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    train_dataset = load_dataset(local_dataset_path, "de-en", split="train")
    train_dataset = tokenize_dataset(train_dataset)
    train_dataset = remove_long_data(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader


def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    labels = [example["labels"] for example in batch]

    input_ids = torch.tensor(
        [BPE_tokenizer.pad({"input_ids": ids}, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
             for ids in input_ids])
    labels = torch.tensor(
        [BPE_tokenizer.pad({"input_ids": lbls}, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
             for lbls in labels])

    return input_ids, labels


def tokenize_dataset(dataset):
    def tokenize_func(example):
        gtext_tokenized = BPE_tokenizer.encode(example["translation"]["de"])
        etext_tokenized = BPE_tokenizer.encode(example["translation"]["en"])
        return {"input_ids": gtext_tokenized, "labels": etext_tokenized}

    tokenized_dataset = dataset.map(tokenize_func, remove_columns=["translation"])
    return tokenized_dataset


def remove_long_data(ds):
    return ds.filter(lambda example: all(len(example[key]) <= MAX_LENGTH for key in ['input_ids', 'labels']))
