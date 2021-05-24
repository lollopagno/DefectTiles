import torch.nn as nn


def convolutional_layers(in_channel: int,
                         out_channel: int,
                         kernel_size: tuple = (3, 3),
                         strides: tuple = (1, 1),
                         padding: tuple = (1, 1)):
    r"""

    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param strides:
    :param padding:
    :return:
    """

    return nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=strides,
                     padding=padding)


def activation_function():
    r"""

    :return:
    """
    return nn.ReLU(inplace=True)


def pooling(kernel_size: tuple = (2, 2),
            strides: tuple = (2, 2)):
    r"""

    :param kernel_size:
    :param strides:
    :return:
    """
    return nn.MaxPool2d(kernel_size=kernel_size, stride=strides)


def up_layers(size: tuple = (2, 2),
              up_mode: str = "same"):
    r"""

    :param size:
    :param up_mode:
    :return:
    """
    return nn.Upsample(size=size, mode=up_mode)