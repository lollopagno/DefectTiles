from UNet.unet import Unet
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet()
model = model.to(device)

# writer = SummaryWriter("runs/unet_experiment_1")
# x = torch.randn(1, 3, 224, 224)
# writer.add_graph(model, x)
# writer.close()

x = torch.randn(size = (1, 1, 512, 512), dtype = torch.float32).cuda()
with torch.no_grad ():
    out = model (x)

print (f'Out net: {out.shape} ')

summary(model, (1, 512, 512))

