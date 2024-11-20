from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader

VOCAB_SIZE = 30000


def train_tokenizer():
    ds = load_dataset("wmt/wmt14", "de-en", split="test")
    def dataset_iterator(dataset):
        for i in dataset:
            yield i["translation"]["de"]
            yield i["translation"]["en"]

    BPE_tokenizer = Tokenizer(BPE())
    BPE_trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True)
    BPE_tokenizer.train_from_iterator(dataset_iterator(ds),BPE_trainer, length = len(ds))
    BPE_tokenizer.save("BPE_Tokenizer.json")


def load_dataset_and_preprocess():
    train_dataset = load_dataset("wmt/wmt14", "de-en", split="train")
    test_dataset = load_dataset("wmt/wmt14", "de-en", split="test")

    test_dataset = tokenize_dataset(test_dataset)
    print(test_dataset[0])
    print("-----------------")
    print(test_dataset[1])
    train_dataset = tokenize_dataset(train_dataset)

    return train_dataset, test_dataset


def tokenize_dataset(dataset):
    BPE_tokenizer = PreTrainedTokenizerFast(tokenizer_file="BPE_Tokenizer.json")

    def tokenize_func(example):
        gtext_tokenized = BPE_tokenizer.encode(example["translation"]["de"])
        etext_tokenized = BPE_tokenizer.encode(example["translation"]["en"])
        return {
            "input_ids": gtext_tokenized,
            "attention_mask": [1] * len(gtext_tokenized),
            "labels": etext_tokenized
        }

    tokenized_dataset = dataset.map(tokenize_func, remove_columns=["translation"])
    return tokenized_dataset


def create_dataloader():
    train, test = load_dataset_and_preprocess()
    train_loader = DataLoader(train)
    test_loader = DataLoader(test)
    return train_loader, test_loader
