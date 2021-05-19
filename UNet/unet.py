import torch
import torch.nn as nn

CHANNEL_INPUT = 1


def net_block(in_channel, out_channel, kernel_size=(3, 3), strides=(1, 1)):
    r"""
    # TODO documentation
    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param strides:
    :return: blocco di convoluzione
    """

    down_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=strides),
        # TODO padding nei livelli di convoluzione??
        nn.ReLU(),
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=strides),
        nn.ReLU()
    )

    return down_block


class Unet(nn.Module):
    r"""
    U-net class
    """

    def __init__(self, block_filter_count=[64, 128, 256, 512, 1024]):
        r"""
        # TODO documentation
        :param block_filter_count:
        """
        super(Unet, self).__init__()

        # Down sampling path
        self.down_block_1 = net_block(CHANNEL_INPUT, block_filter_count[0])
        self.down_block_2 = net_block(block_filter_count[0], block_filter_count[1])
        self.down_block_3 = net_block(block_filter_count[1], block_filter_count[2])
        self.down_block_4 = net_block(block_filter_count[2], block_filter_count[3])

        # Bottleneck
        self.bottleneck = net_block(block_filter_count[3], block_filter_count[4])

        # Up sampling path
        self.upblock_1 = net_block(block_filter_count[4], block_filter_count[3])
        self.upblock_2 = net_block(block_filter_count[3], block_filter_count[2])
        self.upblock_3 = net_block(block_filter_count[2], block_filter_count[1])
        self.upblock_4 = net_block(block_filter_count[1], block_filter_count[0])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sampling = nn.Upsample(size=(2, 2))

        # Outuput rete
        self.out = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        r"""
        # TODO documentation
        :param x:
        :return:
        """

        # Down sampling path
        c1 = self.down_block_1(x)
        p1 = self.maxpool(c1)

        c2 = self.down_block_2(p1)
        p2 = self.maxpool(c2)

        c3 = self.down_block_3(p2)
        p3 = self.maxpool(c3)

        c4 = self.down_block_4(p3)
        p4 = self.maxpool(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Up sampling path
        u1 = self.up_sampling(bn)
        u1 = torch.cat(u1, c4)
        u1 = self.upblock_1(u1)

        u2 = self.up_sampling(u1)
        u2 = torch.cat(u2, c3)
        u2 = self.upblock_2(u2)

        u3 = self.up_sampling(u2)
        u3 = torch.cat(u3, c2)
        u3 = self.upblock_3(u3)

        u4 = self.up_sampling(u3)
        u4 = torch.cat(u4, c1)
        u4 = self.upblock_4(u4)

        output = self.out(u4)

        return output
