import torch
import torch.nn as nn
from UNet.utility import get_convolutional_layers, \
    get_relu, \
    get_max_pooling, \
    get_sigmoid, \
    get_up_sample


class Down(nn.Module):
    r"""
    # TODO documentation
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        super().__init__()

        self.down_block_1 = get_convolutional_layers(in_channel, out_channel)
        self.down_block_2 = get_convolutional_layers(out_channel, out_channel)
        self.activation = get_relu()
        self.max_pooling = get_max_pooling()

    def forward(self, x):
        out_conv = self.down_block_1(x)
        out_relu = self.activation(out_conv)

        out_conv = self.down_block_2(out_relu)
        out_relu = self.activation(out_conv)

        out_pooling = self.max_pooling(out_relu)
        return out_relu, out_pooling


class Bottleneck(nn.Module):
    r"""
    # TODO documentation
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        super().__init__()

        self.bottleneck_block_1 = get_convolutional_layers(in_channel, out_channel)
        self.bottleneck_block_2 = get_convolutional_layers(out_channel, out_channel)
        self.activation = get_relu()

    def forward(self, x):
        out_conv = self.bottleneck_block_1(x)
        out_relu = self.activation(out_conv)

        out_conv = self.bottleneck_block_2(out_relu)
        out_relu = self.activation(out_conv)

        return out_relu


class Up(nn.Module):
    r"""
    # TODO documentation
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        super().__init__()
        self.up_sample = get_up_sample()
        self.concat = Concat()

    def forward(self, x, skip):
        print(f"After upsample: {x.shape}")
        out_up = self.up_sample(x)
        print(f"Concat out upsample: {out_up.shape}")
        print(f"Concat skip: {skip.shape}")
        self.concat(out_up, skip)


class Concat(nn.Module):
    r"""
    # TODO documentation
    """

    def __init__(self):
        super().__init__()

    def forward(self, layer_1, layer_2):
        out_cat = torch.cat((layer_1, layer_2), dim=1)
        return out_cat


class Out(nn.Module):
    r"""
    # TODO documentation
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: tuple):
        super().__init__()

        self.conv = get_convolutional_layers(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size)
        self.sigmoid = get_sigmoid()

    def forward(self, x: torch.Tensor):
        out_conv = self.conv(x)
        out_sigmoid = self.sigmoid(out_conv)

        return out_sigmoid
