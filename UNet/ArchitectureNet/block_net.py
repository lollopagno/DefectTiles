import torch
import torch.nn as nn
from UNet.ArchitectureNet.utility import get_convolutional_layers, \
    get_relu, \
    get_max_pooling, \
    get_sigmoid, \
    get_up_sample


class Down(nn.Module):
    r"""
    Downsampling block.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        r"""
        :param in_channel: number of input filters.
        :param out_channel: number of output filters.
        """
        super().__init__()

        self.conv_1 = get_convolutional_layers(in_channel, out_channel)
        self.conv_2 = get_convolutional_layers(out_channel, out_channel)
        self.activation = get_relu()
        self.max_pooling = get_max_pooling()

    def forward(self, x):
        r"""
        Forward down block.
        """
        out_conv = self.conv_1(x)
        out_relu = self.activation(out_conv)

        out_conv = self.conv_2(out_relu)
        out_relu = self.activation(out_conv)

        out_pooling = self.max_pooling(out_relu)
        return out_relu, out_pooling


class Bottleneck(nn.Module):
    r"""
    Bottleneck block.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        r"""

        :param in_channel: number of input filters.
        :param out_channel: number of output filters.
        """
        super().__init__()

        self.conv_1 = get_convolutional_layers(in_channel, out_channel)
        self.conv_2 = get_convolutional_layers(out_channel, out_channel)
        self.activation = get_relu()

    def forward(self, x):
        r"""
        Forward bottleneck block.
        """
        out_conv = self.conv_1(x)
        out_relu = self.activation(out_conv)

        out_conv = self.conv_2(out_relu)
        out_relu = self.activation(out_conv)

        return out_relu


class Up(nn.Module):
    r"""
    Upsampling block.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        r"""
        :param in_channel: number of input filters.
        :param out_channel: number of output filters.
        """
        super().__init__()

        self.up_sample = get_up_sample()
        self.concat = Concat()
        self.activation = get_relu()
        self.conv_1 = get_convolutional_layers(in_channel=in_channel, out_channel=out_channel)
        self.conv_2 = get_convolutional_layers(out_channel, out_channel)

    def forward(self, x, skip):
        r"""
        Forward upblock.
        :param x: input layer.
        :param skip: skip connection layer to concatenate.
        """
        out_up = self.up_sample(x)
        out_concat = self.concat(out_up, skip)
        out_conv = self.conv_1(out_concat)
        out_relu = self.activation(out_conv)
        out_conv = self.conv_2(out_relu)
        out_relu = self.activation(out_conv)

        return out_relu


class Concat(nn.Module):
    r"""
    Concatenate class.
    Concatenate two levels of the network.
    """

    def __init__(self):
        super().__init__()

    def forward(self, layer_1, layer_2):
        out_concat = torch.cat((layer_1, layer_2), dim=1)
        return out_concat


class Out(nn.Module):
    r"""
    Output block.
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: tuple):
        r"""
        :param in_channel: number of input filters.
        :param out_channel: number of output filters.
        :param kernel_size: kernel size.
        """
        super().__init__()

        self.conv = get_convolutional_layers(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                             padding=(0, 0))
        self.sigmoid = get_sigmoid()

    def forward(self, x: torch.Tensor):
        r"""
        Forward output block.
        """
        out_conv = self.conv(x)
        out_sigmoid = self.sigmoid(out_conv)

        return out_sigmoid
