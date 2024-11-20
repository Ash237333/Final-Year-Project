import os.path
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader


GERMAN_FILEPATH = "German_text.txt"
ENGLISH_FILEPATH = "English_text.txt"
VOCAB_SIZE = 30000

def save_dataset_as_text():
    ds = load_dataset("wmt/wmt14", "de-en", split="test")
    with open(GERMAN_FILEPATH, 'w', encoding='utf-8') as german_file, open(ENGLISH_FILEPATH, 'w',
                                                                           encoding='utf-8') as english_file:
        for x in tqdm(ds):
            gtext = x["translation"]["de"]
            etext = x["translation"]["en"]
            if not gtext.endswith((".", "?", "!", ")")):
                gtext += "."
            if not etext.endswith((".", "?", "!", ")")):
                etext += "."
            english_file.write(etext + " ")
            german_file.write(gtext + " ")

def train_tokenizer():
    if not (os.path.exists(GERMAN_FILEPATH) and os.path.exists(ENGLISH_FILEPATH)):
        save_dataset_as_text()

    BPE_tokenizer = Tokenizer(BPE())
    BPE_trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True)
    BPE_tokenizer.train([ENGLISH_FILEPATH], BPE_trainer)
    BPE_tokenizer.save("BPE_Tokenizer.json")

def load_dataset_and_preprocess():
    train_dataset = load_dataset("wmt/wmt14", "de-en", split="train")
    test_dataset = load_dataset("wmt/wmt14", "de-en", split="test")

    train_dataset = tokenize_dataset(train_dataset)
    test_dataset = tokenize_dataset(test_dataset)

    return train_dataset, test_dataset

def tokenize_dataset(dataset):
    BPE_tokenizer = Tokenizer(BPE())
    BPE_tokenizer.from_file("BPE_Tokenizer.json")

    def tokenize_func(pair):
        gtext_tokenized = BPE_tokenizer.encode(pair["translation"]["de"]).ids
        etext_tokenized = BPE_tokenizer.encode(pair["translation"]["en"]).ids
        return {
            "input_ids": gtext_tokenized,
            "attention_mask": [1] * len(gtext_tokenized),
            "labels": etext_tokenized
        }

    tokenized_dataset = dataset.map(tokenize_func, remove_columns=["translation"])
    return tokenized_dataset
