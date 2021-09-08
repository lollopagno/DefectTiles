from UNet.ArchitectureNet.unet import get_model
from UNet.DatasetTiles.dataset import DatasetTiles, train_test_split
from UNet.training import training_loop, testing_net
from UNet.plot import plot_history, sample_dataset, plot_test_results
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
import time
import datetime
import os
from torchsummary import summary

SHOW_SAMPLES_TRAIN = False
SHOW_SUMMARY = False
TRAIN_NET = True
SHUTDOWN = False

PARENT_MODELS_DIR = "UNet/ModelSaved"
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

# Concatenate all datasets loaded
training_dataset, validation_dataset, test_dataset = ConcatDataset(train_arr), ConcatDataset(valid_arr), \
                                                     ConcatDataset(test_arr)

print(f"** Dataset loaded correctly! Imgs: {len(training_dataset) + len(validation_dataset) + len(test_dataset)} **\n")
print(f"Size train: {len(training_dataset)} - 70%")
print(f"Size validation: {len(validation_dataset)} - 20%")
print(f"Size test: {len(test_dataset)} - 10%")
print(f"Total imgs splitted: {len(training_dataset) + len(validation_dataset) + len(test_dataset)}\n\n")

batch_size = 4

# Training set
training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

# Validation set
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Show samples train dataset
if SHOW_SAMPLES_TRAIN:
    sample_dataset(data_loader=training_loader, batch_size=batch_size)

model = get_model(num_classes)
model = model.to(torch.device("cuda"))

x = torch.randn(size=(1, 3, 256, 256), dtype=torch.float32).cuda()
with torch.no_grad():
    out = model(x)

print(f'Shape out net: {out.shape} ')

if SHOW_SUMMARY:
    summary(model, (3, 256, 256))

if TRAIN_NET:
    num_epochs = 100
    initial_lr = 0.001
    criterion = nn.BCELoss()  # Binary cross-entropy
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - (1 / num_epochs))

    start_time = time.time()
    current_date_hour = datetime.datetime.now()
    new_dir = f"{current_date_hour.year}{current_date_hour.month}{current_date_hour.day}-" \
              f"{current_date_hour.hour}{current_date_hour.minute}{current_date_hour.second}"

    try:
        # Training model
        loss_train, loss_valid, accuracy_valid, IoU_valid, epochs = training_loop(model=model,
                                                                                  num_epochs=num_epochs,
                                                                                  optimizer=optimizer,
                                                                                  lr_scheduler=lr_scheduler,
                                                                                  loss_fn=criterion,
                                                                                  training_loader=training_loader,
                                                                                  validation_loader=validation_loader,
                                                                                  directory=new_dir)
        end_time = time.time()
        print(f"** Training time: {round(((end_time - start_time) / 60), 3)} minutes **\n\n")

        # Show loss and accuracy
        plot_history(loss_train=loss_train, loss_valid=loss_valid, accuracy_valid=accuracy_valid, IoU_valid=IoU_valid,
                     num_epochs=epochs)

        # Test
        test_images, test_masks, test_predicted = testing_net(test_loader=test_loader, model=model, loss_fn=criterion)
        plot_test_results(test_images, test_masks, test_predicted, len(test_images) - 2)

    except Exception as e:
        with open(PARENT_MODELS_DIR + "/" + new_dir + "/log/log.txt", 'a') as f:
            f.write(f'Exception: {e}\n\n')

if SHUTDOWN:
    os.system("shutdown /s /t 1")
