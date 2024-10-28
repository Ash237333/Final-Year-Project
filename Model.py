from torch import nn
from Dataloader import VOCAB_SIZE
import Layers
from Layers import Encoder_Layer, EMBEDDING_DIMENSION, Decoder_Layer

NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

class Transformer(nn.Module):
    def __init__(self):
        """
        Defines all layers to be used in the neural network
        """
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embedding_dim=EMBEDDING_DIMENSION)
        self.encoder_layer_stack = nn.ModuleList([Encoder_Layer() for _ in range(NUM_ENCODER_LAYERS)])
        self.decoder_layer_stack = nn.ModuleList([Decoder_Layer() for _ in range(NUM_DECODER_LAYERS)])


    def forward(self, x):
        """
        Defines the order which tensors are passed through the layers in the neural network

        :param x: Batch of sequences of int labels, (Seq_length, Batch_size)
        :return: The outputted tensor
        """

        #Turns int labels into vector embeddings with pos info
        embedded_vectors = self.embed(x)
        embedded_vectors = Layers.positional_encoder(embedded_vectors)
        encoder_output = embedded_vectors

        #Encoder stack
        for layer in self.encoder_layer_stack:
            encoder_output = layer(encoder_output)

        #Decoder Stack
        for layer in self.decoder_layer_stack:
            encoder_output = layer(embedded_vectors, encoder_output)

        return encoder_output
