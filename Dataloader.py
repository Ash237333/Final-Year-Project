from cProfile import label

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers.models.gpt_neox.modeling_gpt_neox import attention_mask_func

VOCAB_SIZE = 30000
BATCH_SIZE = 100


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
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def collate_fn(batch):
    input_ids, attention_mask, labels = [], [], []
    for i in batch:
        input_ids.append(i["input_ids"])
        attention_mask.append(i["attention_mask"])
        labels.append(i["labels"])

    padded_input_ids = pad_sequence(input_ids, batch_first=True)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }