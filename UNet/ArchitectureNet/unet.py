import torch
import torch.nn as nn
from UNet.ArchitectureNet import block_net as Block


def get_model(num_classes):
    """
    Get the model.
    :param num_classes: number of classes.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(n_classes_out=num_classes)
    model = model.to(device)

    return model


class Unet(nn.Module):
    r"""
    U-net class.
    """

    def __init__(self,
                 n_classes_out: int,
                 block_filter_count: list[int] = [3, 64, 128, 256, 512, 1024]):
        r"""
        Builder of the class.
        :param n_classes_out: number of classes of the problem.
        :param block_filter_count: number of filters for each convolutional layer.
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
            block = Block.Up(in_channel=block_filter_count[5 - i] + block_filter_count[5 - (i + 1)],
                             out_channel=block_filter_count[5 - (i + 1)])
            print(
                f"Block ups {i + 1}-layer: in: {block_filter_count[5 - i] + block_filter_count[5 - (i + 1)]}, out: {block_filter_count[5 - (i + 1)]}")
            self.blocks_up.append(block)

        # Output layer
        print("\n** Out net **")

        self.out = Block.Out(in_channel=block_filter_count[1], out_channel=n_classes_out, kernel_size=(1, 1))
        print(f"Out layer: in: {block_filter_count[1]}, out: {n_classes_out}\n")

        self.blocks_down = nn.ModuleList(self.blocks_down)
        self.bottleneck = nn.ModuleList([self.bottleneck])
        self.blocks_up = nn.ModuleList(self.blocks_up)

    def forward(self, x: torch.Tensor):
        r"""
        Forward of the network.
        :param x: input net.
        :return: output net.
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
            x = block(x, encoder[len(encoder) - (index + 1)])

        # Output layer
        out_net = self.out(x)

        return out_net
