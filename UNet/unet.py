import torch
import torch.nn as nn
from UNet import block_net as Block


# TODO Dubbi:
#  1- Dimensioni delle immagini del dataset
#  2- Preprocessing delle immagini

class Unet(nn.Module):
    r"""
    U-net class
    """

    def __init__(self,
                 block_filter_count: list[int] = [1, 64, 128, 256, 512, 1024]):
        r"""
        # TODO documentation
        :param block_filter_count:
        """
        super(Unet, self).__init__()

        self.blocks_down = []
        self.blocks_up = []

        # Down sampling path
        print("** Block downs **")
        for i in range(0, 4):
            block = Block.Down(in_channel=block_filter_count[i], out_channel=block_filter_count[i + 1])
            print(f"Block down {i + 1}-layer: in: {block_filter_count[i]}, out: {block_filter_count[i + 1]}")
            self.blocks_down.append(block)

        # Bottleneck
        print("\n** Bottleneck **")
        self.bottleneck = Block.Bottleneck(in_channel=block_filter_count[4], out_channel=block_filter_count[5])
        print(f"Bottleneck layer: in: {block_filter_count[4]}, out: {block_filter_count[5]}")

        # Up sampling path
        print("\n** Block ups **")
        for i in range(0, 4):
            block = Block.Up(in_channel=block_filter_count[5 - i], out_channel=block_filter_count[5 - (i + 1)])
            print(f"Block ups {i + 1}-layer: in: {block_filter_count[5 - i]}, out: {block_filter_count[5 - (i + 1)]}")
            self.blocks_up.append(block)

        # Output net
        print("\n** Out net **")
        self.out = Block.Out(in_channel=block_filter_count[1], out_channel=1, kernel_size=(1, 1))
        print(f"Out layer: in: {block_filter_count[1]}, out: {1}\n")

        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.bottleneck = nn.ModuleList([self.bottleneck])
        self.blocks_up = nn.ModuleList(self.blocks_up)

    def forward(self, x: torch.Tensor):
        r"""
        # TODO documentation
        :param x:
        :return:
        """

        encoder = []

        # Down sampling path
        for block in self.blocks_down:
            out_conv, out_pool = block(x)

            encoder.append(out_conv)
            x = out_pool

        out_down_blocks = x

        # Bottleneck
        bn = self.bottleneck[0](out_down_blocks)

        x = bn

        # Up sampling path
        for index, block in enumerate(self.blocks_up):
            print(f"Index: {index}")
            print(f"Index block down: {len(encoder) - (index + 1)}")
            x = block(x, encoder[len(encoder) - (index + 1)])

        # Out layer net
        # out_net = self.out(x)

        return x
