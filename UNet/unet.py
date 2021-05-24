import torch.nn as nn
from UNet import block_net as Block
from UNet.utility import pooling


# TODO Dubbi:
#  1- Dimensioni delle immagini del dataset
#  2- Preprocessing delle immagini

class Unet(nn.Module):
    r"""
    U-net class
    """

    def __init__(self,
                 block_filter_count=[0, 64, 128, 256, 512, 1024]):
        r"""
        # TODO documentation
        :param block_filter_count:
        """
        super(Unet, self).__init__()

        self.blocks_down = []
        self.blocks_up = []

        # Down sampling path
        for i in range(0, 4):
            block = Block.Down(in_channel = block_filter_count[i], out_channel = block_filter_count[i + 1])
            self.blocks_down.append(block)

        # Bottleneck
        self.bottleneck = Block.Bottlneck(in_channel=block_filter_count[3], out_channel=block_filter_count[4])

        # Up sampling path
        # self.upblock_1 = conv_block_transpose(block_filter_count[4], block_filter_count[3])
        # self.upblock_2 = conv_block_transpose(block_filter_count[3], block_filter_count[2])
        # self.upblock_3 = conv_block_transpose(block_filter_count[2], block_filter_count[1])
        # self.upblock_4 = conv_block_transpose(block_filter_count[1], block_filter_count[0])

        self.maxpool = pooling()

        # Outuput rete
        # self.out = nn.Sequential(
        #     nn.Conv2d(in_channels=block_filter_count[0], out_channels=1, kernel_size=(1, 1)),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        r"""
        # TODO documentation
        :param x:
        :return:
        """

        encoder = []
        decoder = []

        # Down sampling path
        for block in self.blocks_down:
            conv = block(x)
            pool = self.maxpool(conv)

            encoder.append(conv)
            x = pool

        # Bottleneck
        bt = self.bottleneck(x)

        # Up sampling path
        # u1 = self.up_sampling(bt)
        #
        # print(f"Bt: {bt.size()}, Type: {bt.dtype}")
        # print(f"U1: {u1.size()}, Type: {u1.dtype}")
        # print(f"C4: {conv_4.size()}, Type: {conv_4.dtype}")
        #
        # u1 = torch.cat((u1, conv_4))
        # u1 = self.upblock_1(u1)
        #
        # u2 = self.up_sampling(u1)
        # u2 = torch.cat((u2, conv_3))
        # u2 = self.upblock_2(u2)
        #
        # u3 = self.up_sampling(u2)
        # u3 = torch.cat((u3, conv_2))
        # u3 = self.upblock_3(u3)
        #
        # u4 = self.up_sampling(u3)
        # u4 = torch.cat((u4, conv_1))
        # u4 = self.upblock_4(u4)
        #
        # output = self.out(u4)

        return bt
