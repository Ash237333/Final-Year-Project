from torch import nn
from Dataloader import VOCAB_SIZE
import Layers
from Layers import Encoder_Layer, EMBEDDING_DIMENSION

NUM_ENCODER_LAYERS = 6

class Seq2seq(nn.Module):
    def __init__(self):
        """
        Defines all layers to be used in the neural network
        """
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embedding_dim=EMBEDDING_DIMENSION)
        self.encoder_layer_stack = nn.ModuleList([Encoder_Layer() for _ in range(NUM_ENCODER_LAYERS)])


    def forward(self, x):
        """
        Defines the order which tensors are passed through the layers in the neural network

        :param x: Batch of sequences of int labels, (Seq_length, Batch_size)
        :return: The outputted tensor
        """

        #Turns sequence of int labels into vector encodings + positional information

        embedded_vectors = self.embed(x)
        pos_added_vectors = Layers.positional_encoder(embedded_vectors)

        for layer in self.encoder_layer_stack:
            pos_added_vectors = layer(pos_added_vectors)

        return pos_added_vectors
