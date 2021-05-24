import torch.nn as nn
from UNet.utility import convolutional_layers
from UNet.utility import activation_function


class Down(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        super().__init__()

        self.down_block_1 = convolutional_layers(in_channel, out_channel)
        self.down_block_2 = convolutional_layers(out_channel, out_channel)
        self.activation = activation_function()

    def forward(self, x):
        conv_1 = self.down_block_1(x)
        conv_1 = self.activation(conv_1)

        conv_2 = self.down_block_2(conv_1)
        conv_2 = self.activation(conv_2)

        return conv_2


class Bottlneck(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        super().__init__()

        self.down_block_1 = convolutional_layers(in_channel, out_channel)
        self.down_block_2 = convolutional_layers(out_channel, out_channel)
        self.activation = activation_function()

    def forward(self, x):
        conv_1 = self.down_block_1(x)
        conv_1 = self.activation(conv_1)

        conv_2 = self.down_block_2(conv_1)
        conv_2 = self.activation(conv_2)

        return conv_2