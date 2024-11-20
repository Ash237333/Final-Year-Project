import Dataloader
from Model import Transformer
import torch
from torch.optim.adam import Adam

EPOCHS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_loader, test_loader = Dataloader.create_dataloader()


def train_one_epoch():
    for i, data in enumerate(train_loader):
        german, english = data
        model.zero_grad()
        output = model(german)
        loss = loss_fn(output, english)
        loss.backward()
        optimizer.step()
    return


def train():
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_one_epoch()
    return

model = Transformer()
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

train()