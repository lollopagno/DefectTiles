import torch
import numpy as np

from UNet.ArchitectureNet.unet import get_model
from UNet.metric import accuracy, IoU
from UNet.earlyStopping import EarlyStopping
import os
from tqdm import tqdm

SHOW_EVERY = 1
PARENT_DIR = "UNet/ModelSaved"


def _round(values):
    return round(values, 3)


def training_loop(model, num_epochs, optimizer, lr_scheduler, loss_fn, training_loader, validation_loader, directory):
    r"""
    Network training.
    :param model: neural network model.
    :param num_epochs: number of epochs.
    :param optimizer: optimize used.
    :param lr_scheduler: scheduler used.
    :param loss_fn: loss function used.
    :param training_loader: training data.
    :param validation_loader: validation data.
    :param directory: name directory.
    """

    print("\n** Training **\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder = create_directory(directory)
    with open(PARENT_DIR + "/" + directory + '/log/log.txt', 'w') as f:
        f.write(
            f'Training:\nEpochs: {num_epochs},\noptimizer: {optimizer.__class__.__name__},\n'
            f'Loss: {loss_fn.__class__.__name__},\nLearning Rate: {optimizer.defaults["lr"]}')

    # Initialized params before training
    # Loss
    training_loss_arr = []
    validation_loss_arr = []

    # Accuracy
    training_accuracy_arr = []
    validation_accuracy_arr = []

    # Intersection over Union (IoU)
    training_IoU_arr = []
    validation_IoU_arr = []

    # Parameters to plot loss and accuracy
    plot_train_loss = []
    plot_validate_loss = []
    plot_validate_accuracy = []
    plot_validate_IoU = []

    # Early stopping
    early_stopping = EarlyStopping()

    for epoch in range(0, num_epochs):
        # Training loop
        model.train()

        training_loss_batch = []
        training_accuracy_batch = []
        training_IoU_batch = []

        train_bar = tqdm(training_loader, total=len(training_loader))

        for X, y in train_bar:
            # Split the train data from the labels
            X = X.to(device)
            y = y.to(device)
            y[y > 0] = 1

            optimizer.zero_grad()

            # Forward pass
            y_predicted = model(X)

            loss = loss_fn(y_predicted, y)
            loss_value = loss.item()

            accuracy_value = accuracy(y_predicted.cpu().detach().numpy(),
                                      y.cpu().detach().numpy())

            IoU_value = IoU(y_predicted.cpu().detach().numpy(),
                            y.cpu().detach().numpy())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Added Loss
            training_loss_batch.append(loss_value)

            # Added accuracy
            training_accuracy_batch.append(accuracy_value)

            # Added IoU
            training_IoU_batch.append(IoU_value)

            train_bar.set_description(
                f"Epoch: {epoch + 1}/{num_epochs}, Loss: {_round(loss_value)}, Acc: {_round(accuracy_value)}, IoU:{_round(IoU_value)}")

        training_loss_arr.append(np.mean(training_loss_batch))
        training_accuracy_arr.append(np.mean(training_accuracy_batch))
        training_IoU_arr.append(np.mean(training_IoU_batch))

        # Set loss (training) for each epoch
        plot_train_loss.append(training_loss_arr[-1])

        model.eval()

        # Validation
        with torch.no_grad():

            validation_loss_batch = []
            validation_accuracy_batch = []
            validation_IoU_batch = []

            # Validation loop
            valid_bar = tqdm(validation_loader, total=len(validation_loader))
            for X, y in valid_bar:
                # Split the train data from the labels
                X = X.to(device)
                y = y.to(device)
                y[y > 0] = 1

                y_predicted = model(X)
                loss = loss_fn(y_predicted, y)
                loss_value = loss.item()

                accuracy_value = accuracy(y_predicted.cpu().detach().numpy(),
                                          y.cpu().detach().numpy())

                IoU_value = IoU(y_predicted.cpu().detach().numpy(),
                                y.cpu().detach().numpy())

                # Added Loss
                validation_loss_batch.append(loss_value)

                # Added accuracy
                validation_accuracy_batch.append(accuracy_value)

                # Added IoU
                validation_IoU_batch.append(IoU_value)

                valid_bar.set_description(
                    f"Epoch: {epoch + 1}, Loss: {_round(loss_value)}, Acc: {_round(accuracy_value)}, IoU:{_round(IoU_value)}")

            validation_loss_arr.append(np.mean(validation_loss_batch))
            validation_accuracy_arr.append(np.mean(validation_accuracy_batch))
            validation_IoU_arr.append(np.mean(validation_IoU_batch))

            # Set loss and accuracy (validation) for each epoch
            plot_validate_loss.append(validation_loss_arr[-1])
            plot_validate_accuracy.append(validation_accuracy_arr[-1])
            plot_validate_IoU.append(validation_IoU_arr[-1])

            if lr_scheduler is not None:
                last_lr = lr_scheduler.get_last_lr()
            else:
                last_lr = optimizer.defaults['lr']

            print(
                f"\n############### Epoch: {epoch + 1}, Train Loss: {_round(training_loss_arr[-1])}, Learning rate: {last_lr},"
                f" Train Acc: {_round(training_accuracy_arr[-1])}, Train IOU: {_round(training_IoU_arr[-1])} ###############")

            print(
                f"############### Epoch: {epoch + 1}, Valid Loss: {_round(validation_loss_arr[-1])}, Learning rate: {last_lr},"
                f" Valid Acc: {_round(validation_accuracy_arr[-1])}, Valid IOU: {_round(validation_IoU_arr[-1])} ###############")

        if epoch % 10 == 0:
            # Save model each n epochs
            save_model(model, epoch, optimizer, training_loss_arr, validation_loss_arr, training_accuracy_arr,
                       validation_accuracy_arr, folder, f"model_epoch_{epoch + 1}.pth")

        # Early Stopping
        early_stopping(np.round(np.mean(validation_loss_arr), 4))
        if early_stopping.early_stop or epoch == num_epochs - 1:
            save_model(model, epoch, optimizer, training_loss_arr, validation_loss_arr, training_accuracy_arr,
                       validation_accuracy_arr, folder, f"best_model_{epoch + 1}.pth")

            # Stop training
            break

        if lr_scheduler is not None:
            lr_scheduler.step()

    return plot_train_loss, plot_validate_loss, plot_validate_accuracy, plot_validate_IoU


def test(test_loader, model, loss_fn):
    r"""
    Show test results of the neural network.
    :param test_loader: test data.
    :param model: net model.
    :param loss_fn: loss function used.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss_arr = []
    test_accuracy_arr = []
    test_IoU_arr = []

    # Parameters to plot the results
    plot_img = []
    plot_mask = []
    plot_predicted = []

    model.eval()

    # Test loop
    with torch.no_grad():
        # Test steps
        test_loss_batch = []
        test_accuracy_batch = []
        test_IoU_batch = []

        test_bar = tqdm(test_loader, total=len(test_loader))

        for X, y in test_bar:
            # Split the train data from the labels
            X = X.to(device)
            y = y.to(device)
            y[y > 0] = 1

            y_predicted = model(X)

            plot_img.append(X)
            plot_mask.append(y)
            plot_predicted.append(y_predicted)

            loss = loss_fn(y_predicted, y)
            loss_value = loss.item()
            accuracy_value = accuracy(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
            IoU_value = IoU(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())

            # Added Loss
            test_loss_batch.append(loss_value)

            # Added Accuracy
            test_accuracy_batch.append(accuracy_value)

            # Added IoU
            test_IoU_batch.append(IoU_value)

        test_loss_arr.append(np.mean(test_loss_batch))
        test_accuracy_arr.append(np.mean(test_accuracy_batch))
        test_IoU_arr.append(np.mean(test_IoU_batch))

        print(
            f"\n############### Test Loss: {test_loss_arr[-1]}, Test Acc: {test_accuracy_arr[-1]}, Test "
            f"IOU: {test_IoU_arr[-1]} ###############")

    return plot_img, plot_mask, plot_predicted


def save_model(model, epoch, optimizer, training_loss, validation_loss, training_acc, validation_acc, new_dir,
               name_file):
    r"""
    Save the model.
    :param model: model to saved.
    :param epoch: current epoch.
    :param optimizer: net optimizer.
    :param training_loss: training loss.
    :param training_acc: training accuracy.
    :param validation_loss: validation loss.
    :param validation_acc: validation accuracy.
    :param new_dir: new directory created.
    :param name_file: name file to be saved. (Extension file .pth)
    """

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Training Loss': np.mean(training_loss),
        'Validation Loss': np.mean(validation_loss),
        'Training Accuracy': np.mean(training_acc),
        'Validation Accuracy': np.mean(validation_acc),
        "directory": new_dir
    }

    torch.save(checkpoint, PARENT_DIR + "/" + new_dir + "/" + name_file)
    print(f"[{epoch + 1}] Model saved!\n")


def load_model(model, optimizer, path):
    """
    Load model.

    :param model: model net.
    :param optimizer: optimizer.
    :param path: path to load model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        file = torch.load(path)
    else:
        file = torch.load(path, map_location='cpu')

    model.state_dict(file['state_dict'])
    optimizer.load_state_dict(file['optimizer_state_dict'])
    epochs = file['epoch']
    # directory = file['directory']

    return model, optimizer, epochs


def create_directory(new_dir):
    r"""
    Create directory for saved model.

    :param new_dir: name directory.
    """

    try:
        os.mkdir(PARENT_DIR)
    except:
        pass

    path = PARENT_DIR + "/" + new_dir
    os.mkdir(path)
    os.mkdir(path + "/log")

    open(path + "/log/log.txt", "x")
    print("Directory and file created!\n")

    return new_dir
