#Imports
from Model import Transformer
import torch
import torch.nn as nn
import Dataloader

#Set up device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load up the model
checkpoint = torch.load("E:/Downloaded_Output_Files_From_Barkla/saves/run26/epoch_10.pth")
model = Transformer()
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

#Prepare the input phrase
input_phrase = "Eines Tages wollt ich in den Club fahr'n."
BPE_tokenizer = Dataloader.load_BPE()
input_tensor = BPE_tokenizer.encode(input_phrase).ids
print(BPE_tokenizer.decode(input_tensor))
input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(device)

target = torch.tensor([[]]).long().to(device) # Start with BOS token (shape: (1, batch_size))

# Step 3: Perform inference (generate tokens one by one)
output_tokens = []
for _ in range(70):
    # Pass the current input and target to the model
    logits = model(input_tensor, target)

    next_token = logits[:, -1, :].argmax(dim=-1)  # Get the next token

    # Append the token to the sequence
    output_tokens.append(next_token.item())

    if next_token == 3:
        break

    target = torch.cat([target, next_token.unsqueeze(0)], dim=1)

phrase = Dataloader.decode_single_phrase(output_tokens)
print(phrase)
print(output_tokens)