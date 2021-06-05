from UNet.ArchitectureNet.unet import Unet
from UNet.DatasetTiles.dataset import DatasetTiles, train_test_split
from UNet.training import training_loop, test
from UNet.plot import plot_history, sample_dataset, plot_test_results
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import time
from torchsummary import summary

SHOW_SAMPLES_TRAIN = False
SHOW_SUMMARY = False

parent_dir = "UNet/DatasetTiles"
BLOWHOLE = "MT_Blowhole"
BREAK = "MT_Break"
CRACK = "MT_Crack"
FRAY = "MT_Fray"
FREE = "MT_Free"
UNEVEN = "MT_Uneven"
defects = [BLOWHOLE, BREAK, CRACK, FRAY, FREE, UNEVEN]
datasets = []

n_classes = 1

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

# Validation set
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Show samples train dataset
if SHOW_SAMPLES_TRAIN:
    sample_dataset(data_loader=training_loader, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Unet(n_classes_out=n_classes)
model = model.to(device)

x = torch.randn(size=(1, 3, 256, 256), dtype=torch.float32).cuda()
with torch.no_grad():
    out = model(x)

print(f'Shape out net: {out.shape} ')

if SHOW_SUMMARY:
    summary(model, (3, 256, 256))

num_epochs = 1  # 100
criterion = nn.BCELoss()  # Binary cross-entropy
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.0001)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.8)

start_time = time.time()

# Training model
loss_train, loss_valid, accuracy_valid = training_loop(model=model,
                                                       num_epochs=num_epochs,
                                                       optimizer=optimizer,
                                                       lr_scheduler=lr_scheduler,
                                                       loss_fn=criterion,
                                                       training_loader=training_loader,
                                                       validation_loader=validation_loader)
end_time = time.time()
print(f"** Training time: {round(((end_time - start_time) / 60), 3)} minutes **\n\n")

# Show loss and accuracy
plot_history(loss_train=loss_train, loss_valid=loss_valid, accuracy_valid=accuracy_valid, num_epochs=num_epochs)

# Test
test_images, test_masks, test_predicted = test(test_loader=test_loader, model=model, loss_fn=criterion)
plot_test_results(test_images, test_masks, test_predicted, len(test_images) - 2)
