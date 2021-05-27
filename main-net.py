from UNet.unet import Unet
from UNet.DatasetTiles.dataset import DatasetTiles, train_test_split
from UNet.metric import accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

parent_dir = "UNet/DatasetTiles"
BLOWHOLE = "MT_Blowhole"
BREAK = "MT_Break"
CRACK = "MT_Crack"
FRAY = "MT_Fray"
FREE = "MT_Free"
UNEVEN = "MT_Uneven"
defects = [BLOWHOLE, BREAK, CRACK, FRAY, FREE, UNEVEN]
datasets = []

# Loaded dataset
print("Loading dataset in progress ...")
for defect in defects:
    datasets.append(DatasetTiles(parent_dir, defect))

dataset = ConcatDataset(datasets)
print(f"** Dataset loaded correctly! Imgs: {len(dataset)} **\n")

training_dataset, validation_dataset, test_dataset = train_test_split(dataset)
print(f"Size train: {len(training_dataset)}")
print(f"Size validation: {len(validation_dataset)}")
print(f"Size test: {len(test_dataset)}")
print(f"Total after split: {len(training_dataset) + len(validation_dataset) + len(test_dataset)}\n\n")

batch_size = 4

# TODO valutare se splittare il DT in immagini e maschere

# Training set
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

# Validation set
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet()
model = model.to(device)

# writer = SummaryWriter("runs/unet_experiment_1")
# x = torch.randn(1, 3, 224, 224)
# writer.add_graph(model, x)
# writer.close()

x = torch.randn(size=(1, 1, 512, 512), dtype=torch.float32).cuda()
with torch.no_grad():
    out = model(x)

print(f'Shape out net: {out.shape} ')

# summary(model, (1, 512, 512))

# TODO Minibatch = ??
# TODO Iteration for epoch = ??
num_epochs = 0  # 100
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.0001)

# Training loop
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#
#     output = model(x)
#     loss = criterion(output, y)
#     loss.backward()
#     optimizer.step()
#
#     print('Epoch {}, Loss {}'.format(epoch, loss.item()))
