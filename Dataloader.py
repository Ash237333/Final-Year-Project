import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import Batch_Sampler

VOCAB_SIZE = 37000
dataset_path = "wmt/wmt14"
TARGET_TOTAL_TOKENS = 9000


def train_tokenizer():
    ds = load_dataset(dataset_path, "de-en", split="train")
    def dataset_iterator(dataset):
        for i in dataset:
            yield i["translation"]["de"]
            yield i["translation"]["en"]

    BPE_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    BPE_tokenizer.pre_tokenizer = ByteLevel()

    BPE_trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    BPE_tokenizer.train_from_iterator(dataset_iterator(ds),BPE_trainer, length = len(ds)*2)
    BPE_tokenizer.pad_token = "[PAD]"
    BPE_tokenizer.bos_token = "[BOS]"
    BPE_tokenizer.eos_token = "[EOS]"
    BPE_tokenizer.unk_token = "[UNK]"
    BPE_tokenizer.save("BPE_Tokenizer.json")


def create_dataloader():
    test_dataset = load_dataset(dataset_path,"de-en", split="test", download_mode="force_redownload")
    test_dataset = tokenize_dataset(test_dataset)
    test_dataset = remove_long_data(test_dataset)
    test_sampler = Batch_Sampler.Custom_Sampler(test_dataset, TARGET_TOTAL_TOKENS)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

    train_dataset = load_dataset(dataset_path, "de-en", split="train", download_mode="force_redownload")
    train_dataset = tokenize_dataset(train_dataset)
    train_dataset = remove_long_data(train_dataset)
    train_sampler = Batch_Sampler.Custom_Sampler(train_dataset, TARGET_TOTAL_TOKENS)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

    return train_loader, test_loader


def collate_fn(batch):
    input_ids = []
    labels = []
    for example in batch:
        input_id = example["input_ids"]
        label = example["labels"]
        input_ids.append(torch.tensor(input_id))
        labels.append(torch.tensor(label))

    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return input_ids, labels


def tokenize_dataset(dataset):
    BPE_tokenizer = load_BPE()
    def tokenize_func(example):
        gtext_tokenized = BPE_tokenizer.encode(example["translation"]["de"]).ids
        etext_tokenized = BPE_tokenizer.encode(example["translation"]["en"]).ids
        return {"input_ids": gtext_tokenized, "labels": etext_tokenized}

    tokenized_dataset = dataset.map(tokenize_func, remove_columns=["translation"])
    return tokenized_dataset


def decode_single_phrase(token_array):
    BPE_tokenizer = load_BPE()
    decoded_phrase = BPE_tokenizer.decode(token_array)
    array = []
    for i in token_array:
        array.append(BPE_tokenizer.id_to_token(i))
    print(array)
    return decoded_phrase


def remove_long_data(ds):
    return ds.filter(lambda example: all(len(example[key]) <= 8000 for key in ['input_ids', 'labels']))


def load_BPE():
    tokenizer = Tokenizer.from_file("BPE_Tokenizer.json")
    return tokenizer