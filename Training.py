import Dataloader
from Model import Transformer
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from Scheduler import WarmupScheduler
from Layers import EMBEDDING_DIMENSION

EPOCHS = 1


def train_one_epoch(epoch_num):
    running_loss = 0

    for i, data in enumerate(train_loader):
        #Calc current step and update LR
        current_step = i + (epoch_num * len(train_loader) + 1)
        scheduler.update_lr(current_step)

        #Extract data and send to device
        german, english = data
        german, english = german.to(device), english.to(device)

        optimizer.zero_grad()
        output = model(german, english)

        #Flattern output and targets to a form CEL accepts
        flattened_output = output.view(-1, output.shape[2])
        flattened_english = english.view(-1)

        #Calc loss and backpropagate
        loss = loss_fn(flattened_output, flattened_english)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        #Log metrics every 100 mini-batches
        if i % 100 == 99:
            avg_loss = running_loss/100
            running_loss = 0
            writer.add_scalar("Loss/train", avg_loss, current_step)


def train():
    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch + 1, " -------------")
        model.train(True)
        train_one_epoch(epoch)
        eval_model()
    return

def eval_model():
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
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = CrossEntropyLoss()
    train_loader, test_loader = Dataloader.create_dataloader()
    model = Transformer()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, 4000, EMBEDDING_DIMENSION)
    train()
    writer.flush()