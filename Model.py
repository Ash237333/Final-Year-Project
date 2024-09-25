import torch
from torch import nn
from Dataloader import VOCAB_SIZE


class Seq2seq(nn.Module):
    def __init__(self, input_size):
        """
        Defines all layers to be used in the neural network
        """
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embedding_dim=100)

    def forward(self, x):
        """
        Defines the order which tensors are passed through the layers in the neural network

        :param x: The inputted tensor
        :return: The outputted tensor
        """

        embedded_vectors = self.embed(x)
        pos_added_vectors = self.positional_encoder(embedded_vectors)
        return pos_added_vectors

    def positional_encoder(self, input):
        """
        Defines the positional encoder.

        :param input: The inputted tensor after vector embedding
        :return: A tensor with the same shape as the input but with the positional
        encoding added to the third dimension.
        """

        #Initialises the result tensor as an empty tensor of shape (seq_length, embedding_dim)
        seq_length, batch_size, embedding_dim = input.shape
        result = torch.empty(seq_length, embedding_dim)

        #Calculates the denominator taking into account the difference for even and odd embedding positions
        positions = torch.arange(start=0, end=seq_length, step=1, dtype=torch.float).unsqueeze(1)
        embedding_positions = torch.arange(start=0, end=embedding_dim, step=1, dtype=torch.float)
        calculate_i = (embedding_positions // 2) * 2
        divisor = 10000 ** (calculate_i / embedding_dim)

        #Calculates the full term within the sin/cos function
        positions = positions / divisor

        #Applies sin/cos based on the numbers parity
        result[:, ::2] = torch.sin(positions[:, ::2])
        result[:, 1::2] = torch.cos(positions[:, 1::2])

        #Adds the positional encoding to the embedded vectors
        result = input + result.unsqueeze(1)

        return result
