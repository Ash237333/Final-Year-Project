#Imports
from Model import Transformer
import torch
from transformers import PreTrainedTokenizerFast
import torch.nn as nn
import Dataloader

#Set up device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load up the model
checkpoint = torch.load("./saves/run9/epoch_1.pth")
model = Transformer()
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

#Prepare the input phrase
input_phrase = "ich"
BPE_tokenizer = PreTrainedTokenizerFast(tokenizer_file="BPE_Tokenizer.json")
input_tensor = BPE_tokenizer.encode(input_phrase)
print(input_tensor)
input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

target = torch.tensor([[]]).long().to(device) # Start with BOS token (shape: (1, batch_size))

# Step 3: Perform inference (generate tokens one by one)
output_tokens = []
for _ in range(30):
    # Pass the current input and target to the model
    logits = model(input_tensor, target)

    target = logits.argmax(dim=-1)

    pred_tokens_print = target.squeeze()
    print(pred_tokens_print)

phrase = Dataloader.decode_single_phrase(pred_tokens_print)
print(phrase)