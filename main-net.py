from UNet.unet import Unet
from UNet.DatasetTiles.dataset import DatasetTiles, train_test_split
from UNet.metric import accuracy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchsummary import summary
import cv2 as cv

parent_dir = "UNet/DatasetTiles"
BLOWHOLE = "MT_Blowhole"
BREAK = "MT_Break"
CRACK = "MT_Crack"
FRAY = "MT_Fray"
FREE = "MT_Free"
UNEVEN = "MT_Uneven"
defects = [BLOWHOLE, BREAK, CRACK, FRAY, FREE, UNEVEN]
datasets = []

n_classes = len(defects)

# Loaded dataset
print("Loading dataset in progress ...")
for defect in defects:
    datasets.append(DatasetTiles(parent_dir, defect))

dataset = ConcatDataset(datasets)
print(f"** Dataset loaded correctly! Imgs: {len(dataset)} **\n")

training_dataset, validation_dataset, test_dataset = train_test_split(dataset)
print(f"Size train: {len(training_dataset)} - 70%")
print(f"Size validation: {len(validation_dataset)} - 20%")
print(f"Size test: {len(test_dataset)} - 10%")
print(f"Total imgs splitted: {len(training_dataset) + len(validation_dataset) + len(test_dataset)}\n\n")

batch_size = 8

# Training set
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

# Show samples train dataset
# import matplotlib.pyplot as plt
# import numpy as np
#
# for test_images, test_labels in training_loader:
#     sample_image = test_images[2]   # Reshape them according to your needs.
#     sampl1e_label = test_labels[2]
#
#     img = np.squeeze(sample_image)
#     plt.title('Image')
#     plt.imshow(img)
#     plt.show()
#
#     label = np.squeeze(sampl1e_label)
#     plt.title('Label')
#     plt.imshow(label)
#     plt.show()
#     break

# Validation set
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(n_classes=n_classes)
model = model.to(device)

# x = torch.randn(size=(1, 1, 512, 512), dtype=torch.float32).cuda()
# with torch.no_grad():
#     out = model(x)
#
# print(f'Shape out net: {out.shape} ')
#
# summary(model, (1, 512, 512))

# TODO Iteration for epoch = ??
num_epochs = 0  # 100
criterion = nn.BCELoss()  # Binary cross-entropy
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
