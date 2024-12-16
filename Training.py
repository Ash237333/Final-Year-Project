import Dataloader
from Model import Transformer
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import time
from Scheduler import WarmupScheduler
from Layers import EMBEDDING_DIMENSION

EPOCHS = 1


def train_one_epoch(epoch_num):
    running_loss = 0
    for i, data in enumerate(train_loader):
        scheduler.update_lr(i + (epoch_num * len(train_loader) + 1))
        german, english = data
        german, english = german.to(device), english.to(device)
        optimizer.zero_grad()
        output = model(german, english)
        flattened_output = output.view(-1, output.shape[2])
        flattened_english = english.view(-1)
        #Flattened squishes all sentences in the batch to one long string
        #Needed because CEL only works on a 1D target and 2D input
        loss = loss_fn(flattened_output, flattened_english)
        running_loss += loss.item()
        with open("log_file.txt", "a") as file:
            file.write(f"{loss.item()}\n")
        print(loss.item())
        loss.backward()
        optimizer.step()
        if i % 100 == 99:
            end_time = time.time()  # End timing
            running_loss = running_loss / 100.0
            running_loss = 0
    return running_loss


def train():
    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch + 1, " -------------")
        model.train(True)
        train_one_epoch(epoch)
        eval()
    return

def eval():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            german, english = data
            german, english = german.to(device), english.to(device)
            output = model(german, english)
            flattened_output = output.view(-1, output.shape[2])
            flattened_english = english.view(-1)
            loss = loss_fn(flattened_output, flattened_english)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print("Validation Loss: ", avg_loss)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = CrossEntropyLoss()
    train_loader, test_loader = Dataloader.create_dataloader()
    model = Transformer()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, 4000, EMBEDDING_DIMENSION)
    train()