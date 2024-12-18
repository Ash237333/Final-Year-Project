import Dataloader
from Model import Transformer
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from Scheduler import WarmupScheduler
from Layers import EMBEDDING_DIMENSION
from tqdm import tqdm
import os

EPOCHS = 1

torch.manual_seed(1)

SAVE_DIR = "./saves/run1"
os.makedirs(SAVE_DIR)


def train_one_epoch(epoch_num):
    """
    Handles training for the model for a single epoch. Average loss over every 100 mini-batches
    is logged in tensorboard

    :param epoch_num: The number of epochs already completed, used to calculate current step number
    """

    length = len(train_loader)
    running_loss = 0

    for i, data in tqdm(enumerate(train_loader), total=length):
        # Calculate current step and update LR
        optimizer.zero_grad()
        current_step = i + (epoch_num * length + 1)


        # Extract data, send to device and pass through the network
        german, english = data

        german, english = german.to(device), english.to(device)
        output = model(german, english)

        # Flatten output and targets to a form CEL accepts
        output = output.view(-1, output.shape[2])
        english = english.view(-1)

        # Calculate loss and backpropagate
        loss = loss_fn(output, english)
        running_loss += loss.item()
        loss.backward()

        lr = optimizer.step_and_update()
        writer.add_scalar("Learning Rate", lr, current_step)

        #Log metrics every 100 mini-batches
        if i % 100 == 99:
            avg_loss = running_loss/100
            running_loss = 0
            writer.add_scalar("Loss/train", avg_loss, current_step)
            writer.flush()


def train():
    """
    Basic training loop to call training and eval methods for each epoch
    """

    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch + 1, " -------------")
        model.train(True)
        train_one_epoch(epoch)
        eval_model(epoch)
        save_checkpoint(epoch, model, optimizer)
    return

def eval_model(epoch_num):
    """
    Runs through the test dataset and calculates losses but doesn't update.
    The average loss over the entire test dataset is logged in tensorboard

    :param epoch_num: The number of epochs already completed, used to log loss
    """

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            german, english = data
            german, english = german.to(device), english.to(device)

            output = model(german, english)

            # Flatten output and targets to a form CEL accepts
            flattened_output = output.view(-1, output.shape[2])
            flattened_english = english.view(-1)

            # Calculate loss but no backpropagation
            loss = loss_fn(flattened_output, flattened_english)
            total_loss += loss.item()

    # Calculate average loss and log after all batches completed
    avg_loss = total_loss / len(test_loader)
    writer.add_scalar("Loss/test", avg_loss, epoch_num + 1)
    writer.flush()


def save_checkpoint(epoch_num, model, optimizer):
    """
    Save model and optimizer states as a checkpoint.

    :param epoch_num: The epoch number (used in the filename)
    :param model: The model to save
    :param optimizer: The optimizer to save
    """
    checkpoint_path = os.path.join(SAVE_DIR, f"epoch_{epoch_num + 1}.pth")
    torch.save({
        'epoch': epoch_num + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    loss_fn = CrossEntropyLoss()
    train_loader, test_loader = Dataloader.create_dataloader()
    model = Transformer()
    model.to(device)
    optimizer = WarmupScheduler(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 4000, EMBEDDING_DIMENSION)
    train()
    writer.flush()
    writer.close()