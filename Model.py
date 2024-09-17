from torch import nn

class Seq2seq(nn.Module):
    def __init__(self):
        """
        Defines all layers to be used in the neural network
        """
        super().__init__()

    def forward(self, x):
        """
        Defines the order which tensors are passed through the layers in the neural network

        :param x: The inputted tensor
        :return: The outputted tensor
        """
        return x