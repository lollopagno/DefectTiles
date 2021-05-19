from UNet import unet as Net
from torch.utils.tensorboard import SummaryWriter

net = Net.Unet()

writer = SummaryWriter('experiment_unet_1')
