import torch
import torch.nn as nn

from torch.utils.data import ConcatDataset, DataLoader
from UNet.DatasetTiles.dataset import DatasetTiles, train_test_split
from UNet.plot import plot_test_results
from UNet.training import testing_net
from UNet.ArchitectureNet.unet import get_model

PARENT_DATASET_DIR = "UNet/DatasetTiles/"
BLOWHOLE = "MT_Blowhole"
CRACK = "MT_Crack"
FREE = "MT_Free"
defects = [BLOWHOLE, CRACK, FREE]
datasets = []
train_arr = []
valid_arr = []
test_arr = []

num_classes = 1

# Loaded dataset
print("Loading dataset in progress ...")
for defect in defects:
    # Load each type of defect
    datasets.append(DatasetTiles(PARENT_DATASET_DIR, defect))

for dataset in datasets:
    # Split each dataset in train, validation and test set
    training_dataset, validation_dataset, test_dataset = train_test_split(dataset)

    train_arr.append(training_dataset)
    valid_arr.append(validation_dataset)
    test_arr.append(test_dataset)

training_dataset, validation_dataset, test_dataset = ConcatDataset(train_arr), ConcatDataset(valid_arr), \
                                                     ConcatDataset(test_arr)

print(
    f"** Dataset loaded correctly! Imgs: {len(training_dataset) + len(validation_dataset) + len(test_dataset)} **\n")
print(f"Size train: {len(training_dataset)} - 70%")
print(f"Size validation: {len(validation_dataset)} - 20%")
print(f"Size test: {len(test_dataset)} - 10%")
print(f"Total imgs splitted: {len(training_dataset) + len(validation_dataset) + len(test_dataset)}\n\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("UNet/ModelSaved/2021913-91227/best_model_29.pth")  # TODO replace with your path

model = get_model(num_classes)
model.to(device)
model.load_state_dict(checkpoint['state_dict'])
criterion = nn.BCELoss()

# Test set
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

test_images, test_masks, test_predicted = testing_net(test_loader=test_loader, model=model, loss_fn=criterion)
plot_test_results(test_images, test_masks, test_predicted, 4)
