from fastapi import FastAPI
import torch
import torch.nn as nn
from Model import Transformer
from transformers import PreTrainedTokenizerFast
import Dataloader
from fastapi.middleware.cors import CORSMiddleware

#Setup torch, Load in model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
checkpoint = torch.load("./saves/run13/epoch_4.pth")
model = Transformer()
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

#Setup FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/translation/{german_text}")
async def translate(german_text: str):
    BPE_tokenizer = PreTrainedTokenizerFast(tokenizer_file="BPE_Tokenizer.json")
    input_tensor = BPE_tokenizer.encode(german_text)
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

    target = torch.tensor([[]]).long().to(device)

    output_tokens = []
    for _ in range(7):
        # Pass the current input and target to the modelI love you
        logits = model(input_tensor, target)

        next_token = logits[:, -1, :].argmax(dim=-1)  # Get the next token

        # Append the token to the sequence
        output_tokens.append(next_token.item())
        target = torch.cat([target, next_token.unsqueeze(0)], dim=1)

    phrase = Dataloader.decode_single_phrase(output_tokens)


    return phrase