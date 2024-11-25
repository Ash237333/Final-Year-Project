import Dataloader
from Model import Transformer
import torch
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss

EPOCHS = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
loss_fn = CrossEntropyLoss()
train_loader, test_loader = Dataloader.create_dataloader()


def train_one_epoch():
    running_loss = 0
    last_loss = 0

    for i, data in enumerate(train_loader):
        german, english = data
        german, english = german.to(device), english.to(device)
        optimizer.zero_grad()
        output = model(german, english)
        loss = loss_fn(output, english)
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0
    return last_loss


def train():
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1} -------------")
        model.train(True)
        train_one_epoch()
        eval()
    return

def eval():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            german, english = data
            german, english = german.to(device), english.to(device)

            output = model(german)
            loss = loss_fn(output, english)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Validation Loss: {avg_loss:.4f}")


model = Transformer()
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

train()