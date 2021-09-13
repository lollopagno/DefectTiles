import torch.nn as nn


def get_convolutional_layers(in_channel: int,
                             out_channel: int,
                             kernel_size: tuple = (3, 3),
                             strides: tuple = (1, 1),
                             padding: tuple = (1, 1)):
    r"""
    Convolutional layer.
    :param in_channel: number of input filters.
    :param out_channel: number of output filters.
    :param kernel_size: kernel size.
    :param strides: stride size.
    :param padding: padding size.
    """

    return nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=strides,
                     padding=padding)


def get_relu():
    r"""
    Relu activation function.
    """
    return nn.ReLU(inplace=True)


def get_sigmoid():
    r"""
    Sigmoid activation function.
    """
    return nn.Sigmoid()


def get_max_pooling(kernel_size: tuple = (2, 2),
                    strides: tuple = (2, 2)):
    r"""
    Max pooling layer.
    :param kernel_size: kernel size.
    :param strides: stride size.
    """
    return nn.MaxPool2d(kernel_size=kernel_size, stride=strides)


def get_up_sample(scale_factor: int = 2):
    r"""
    Up sample layer.
    :param scale_factor: scale factor.
    """
    return nn.Upsample(scale_factor=scale_factor)


def get_batch_normalization(out_channel: int):
    r"""
    Batch normalization layer.
    :param out_channel: number of output filters.
    """
    return nn.BatchNorm2d(out_channel)
