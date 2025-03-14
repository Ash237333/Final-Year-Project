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
import torch.nn as nn
from torchmetrics.text.bleu import BLEUScore

EPOCHS = 1
ACCUMULATION_STEPS = 8

SAVE_DIR = "./saves/runtest"
os.makedirs(SAVE_DIR)

BPE_tokenizer = Dataloader.load_BPE()

def eos_append(targets, eos_token=3):
    #Adds the eos as the last token in each sequence for loss calc
    #Done this way to prevent storing multiple copies on the gpu
    eos_column = torch.full((targets.shape[0],1), eos_token, device=targets.device)
    targets = torch.cat([targets, eos_column], dim=1)
    return targets


def train_one_epoch(epoch_num):
    """
    Handles training for the model for a single epoch. Average loss over every 100 mini-batches
    is logged in tensorboard

    :param epoch_num: The number of epochs already completed, used to calculate current step number
    """

    length = len(train_loader)
    running_loss = 0

    optimizer.zero_grad()

    for i, data in tqdm(enumerate(train_loader), total=length):
        # Extract data, send to device and pass through the network
        german, english = data

        german, english = german.to(device), english.to(device)
        output = model(german, english)
        english = eos_append(english)

        # Flatten output and targets to a form CEL accepts
        output = output.view(-1, output.shape[2])
        english = english.view(-1)

        # Calculate loss and backpropagate
        loss = loss_fn(output, english) / ACCUMULATION_STEPS
        running_loss += loss.item() * ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == length:
            lr, step = optimizer.step_and_update()
            writer.add_scalar("Learning Rate", lr, step)
            optimizer.zero_grad()

            # Log metrics every 100 mini-accumulated-batches
            if step % 100 == 99:
                avg_loss = running_loss / 100
                running_loss = 0
                writer.add_scalar("Loss/train", avg_loss, step)




def train():
    """
    Basic training loop to call training and eval methods for each epoch
    """

    for epoch in range(EPOCHS):
        print("EPOCH: ",epoch + 1, " -------------")
        model.train(True)
        train_one_epoch(epoch)
        eval_model(epoch)
        writer.flush()
        save_checkpoint(epoch, model, optimizer)
    return

def eval_model(epoch_num):
    """
    Runs through the test dataset and calculates losses but doesn't update.
    The average loss over the entire test dataset is logged in tensorboard

    :param epoch_num: The number of epochs already completed, used to log loss
    """
    all_predictions = []
    all_originals = []

    tokeniser = Dataloader.load_BPE()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            german, english = data
            german, english = german.to(device), english.to(device)

            output = model(german, english)
            english = eos_append(english)

            # Flatten output and targets to a form CEL accepts
            flattened_output = output.view(-1, output.shape[2])
            flattened_english = english.view(-1)

            # Calculate loss but no backpropagation
            loss = loss_fn(flattened_output, flattened_english)
            total_loss += loss.item()

            predicted_ids = output.argmax(dim=-1)
            for ids in predicted_ids:
                predicted_tokens = tokeniser.decode(ids.tolist())
                all_predictions.append(predicted_tokens)

            for ids in english:
                original_tokens = tokeniser.decode(ids.tolist())
                all_originals.append(original_tokens)

    # Calculate average loss and log after all batches completed
    avg_loss = total_loss / len(test_loader)
    writer.add_scalar("Loss/test", avg_loss, epoch_num + 1)

    bleu = BLEUScore()
    bleu_score = bleu(all_predictions, all_originals) * 100
    writer.add_scalar("BLEU/test", bleu_score, epoch_num + 1)


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
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")



if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    loss_fn = CrossEntropyLoss(ignore_index=0)
    train_loader, test_loader = Dataloader.create_dataloader()
    model = Transformer()
    model = nn.DataParallel(model)
    model.to(device)
    optimizer = WarmupScheduler(Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 4000, EMBEDDING_DIMENSION)
    train()
    writer.flush()
    writer.close()