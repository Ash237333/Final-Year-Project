import os.path
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


GERMAN_FILEPATH = "German_text.txt"
ENGLISH_FILEPATH = "English_text.txt"

if not (os.path.exists(GERMAN_FILEPATH) and os.path.exists(ENGLISH_FILEPATH)):
    ds = load_dataset("wmt/wmt14", "de-en", split="test")

    with open(GERMAN_FILEPATH, 'w', encoding='utf-8') as german_file, open(ENGLISH_FILEPATH, 'w', encoding='utf-8') as english_file:
        for x in tqdm(ds):
            gtext = x["translation"]["de"]
            etext = x["translation"]["en"]
            if not gtext.endswith((".", "?", "!", ")")):
                gtext += "."
            if not etext.endswith((".", "?", "!", ")")):
                etext += "."
            english_file.write(etext + " ")
            german_file.write(gtext + " ")

BPE_tokenizer = Tokenizer(BPE())
BPE_trainer = BpeTrainer(vocab_size=30000, show_progress=True)
BPE_tokenizer.train([ENGLISH_FILEPATH],BPE_trainer)
BPE_tokenizer.save("BPE_Tokenizer.json")
