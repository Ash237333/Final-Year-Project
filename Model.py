import torch
from torch import nn
from Dataloader import VOCAB_SIZE
import Layers
from Layers import Encoder_Layer, EMBEDDING_DIMENSION, Decoder_Layer

NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
PAD_VALUE = 0

def pad_mask(input_seq):
    """
    Takes in batched integer seq labels outputs binary mask of padded values.

    :param input_seq: Batched tensor of integer sequence labels (B, seq_length)
    :return: A binary mask with padded values being 1 everything else is 0
    """
    input_seq = input_seq == PAD_VALUE
    return input_seq

def subsequent_mask(target_seq_len, device):
    ones = torch.ones(target_seq_len, target_seq_len, device=device)
    mask = torch.triu(ones, diagonal=1)
    return mask.bool()

class Transformer(nn.Module):
    def __init__(self):
        """
        Defines all layers to be used in the neural network
        """
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embedding_dim=EMBEDDING_DIMENSION)
        self.encoder_layer_stack = nn.ModuleList([Encoder_Layer() for _ in range(NUM_ENCODER_LAYERS)])
        self.decoder_layer_stack = nn.ModuleList([Decoder_Layer() for _ in range(NUM_DECODER_LAYERS)])
        self.fc_final = nn.Linear(EMBEDDING_DIMENSION, VOCAB_SIZE)


    def forward(self, input, target):
        """
        Defines the order which tensors are passed through the layers in the neural network

        :param input: Batch of sequences of int labels, (Seq_length, Batch_size)
        :return: The outputted tensor
        """

        #Set up enc inputs by turning labels into embedded vectors
        #Calculate padding mask for inputs as well
        padded_mask = pad_mask(input)
        x = self.embed(input)
        x = Layers.positional_encoder(x)

        #Encoder stack
        for layer in self.encoder_layer_stack:
            x = layer(x, padded_mask)


        #Set up decoder inputs by embedding target labels
        #Calculate padding mask for targets as well
        target_padding_mask = pad_mask(target)
        target_subsequent_mask = subsequent_mask(target.shape[1], target.device)

        y = self.embed(target)
        y = Layers.positional_encoder(y)


        #Decoder Stack
        for layer in self.decoder_layer_stack:
            y = layer(y, x, padded_mask,target_padding_mask, target_subsequent_mask)

        y = self.fc_final(y)

        return y
