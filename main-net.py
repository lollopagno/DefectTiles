from UNet.unet import Unet
import torch
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Non-trainable params: {(total_params - trainable_params)}")
print("********************************\n\n")

# writer = SummaryWriter("runs/unet_experiment_1")
# x = torch.randn(1, 3, 224, 224)
# writer.add_graph(model, x)
# writer.close()

summary(model, (1, 512, 512))

