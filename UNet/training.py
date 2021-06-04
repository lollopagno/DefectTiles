import torch
import numpy as np
from sklearn.metrics import jaccard_score as jc
from UNet.metric import accuracy
from UNet.earlyStopping import EarlyStopping
import datetime
import os

SHOW_EVERY = 1
PARENT_DIR = "UNet/ModelSaved"


def training_loop(model, num_epochs, optimizer, lr_scheduler, loss_fn, training_loader, validation_loader):
    r"""
    Network training
    :param model: neural network model
    :param num_epochs: number of epochs
    :param optimizer: optimize used
    :param lr_scheduler: scheduler used
    :param loss_fn: loss function used
    :param training_loader: training data
    :param validation_loader: validation data
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialized params before training
    # Loss
    training_loss_arr = []
    validation_loss_arr = []

    # Accuracy
    training_accuracy_arr = []
    validation_accuracy_arr = []

    # Jaccard index
    training_jaccard_arr = []
    validation_jaccard_arr = []

    # Parameters to plot loss and accuracy
    plot_train_loss = np.zeros(num_epochs)
    plot_validate_loss = np.zeros(num_epochs)
    plot_validate_accuracy = np.zeros(num_epochs)

    # Early stopping
    early_stopping = EarlyStopping()

    for epoch in range(0, num_epochs):
        print(f"--> Epoch: {epoch + 1}/{num_epochs}")

        # Training steps
        training_loss_batch = 0.0
        training_accuracy_batch = 0.0
        training_jaccard_batch = 0.0
        num_steps = 0

        for _, batch in enumerate(training_loader):
            torch.cuda.empty_cache()

            # Training model
            model.train()

            # Split the train data from the labels
            X = batch[0].to(device)
            y = batch[1].to(device)
            y[y > 0] = 1

            # Forward pass
            y_predicted = model(X)
            loss = loss_fn(y_predicted, y)

            optimizer.zero_grad()

            # Backward pass
            loss.backward()
            optimizer.step()

            training_loss_batch += loss.item()
            training_accuracy_batch += accuracy(y_predicted.cpu().detach().numpy(),
                                                y.cpu().detach().numpy())

            print(y.shape)
            print(y_predicted.shape)
            training_jaccard_batch += jc(y_true=y.cpu().detach().numpy(), y_pred=y_predicted.cpu().detach().numpy())

            num_steps += 1

        train_loss_for_this_epoch = np.divide(training_loss_batch, num_steps)
        training_loss_arr.append(train_loss_for_this_epoch)
        training_accuracy_arr.append(np.divide(training_accuracy_batch, num_steps))
        training_jaccard_arr.append(np.divice(training_jaccard_batch, num_steps))

        # Set loss (training) for each epoch
        plot_train_loss[epoch] = train_loss_for_this_epoch

        # Validation
        with torch.no_grad():

            num_steps = 0
            validation_loss_batch = 0.0
            validation_accuracy_batch = 0.0
            validation_jaccard_batch = 0.0

            model.eval()

            # Validation steps
            for index, batch in enumerate(validation_loader):
                torch.cuda.empty_cache()

                X = batch[0].to(device)
                y = batch[1].to(device)
                y[y > 0] = 1

                y_predicted = model(X)
                loss = loss_fn(y_predicted, y)
                validation_loss_batch += loss.item()

                validation_accuracy_batch += accuracy(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
                validation_jaccard_batch += jc(y_true=y.cpu().detach().numpy(),
                                               y_pred=y_predicted.cpu().detach().numpy())

                num_steps += 1

            valid_loss_for_this_epoch = np.divide(validation_loss_batch, num_steps)
            valid_accuracy_for_this_epoch = np.divide(validation_accuracy_batch, num_steps)

            validation_loss_arr.append(valid_loss_for_this_epoch)
            validation_accuracy_arr.append(valid_accuracy_for_this_epoch)
            validation_jaccard_arr.append(np.divide(validation_jaccard_batch, num_steps))

            # Set loss and accuracy (validation) for each epoch
            plot_validate_loss[epoch] = train_loss_for_this_epoch
            plot_validate_accuracy[epoch] = valid_accuracy_for_this_epoch

        if epoch % SHOW_EVERY == 0 or epoch == num_epochs - 1:
            print(f"** Training\n\t\tLoss: {np.round(np.mean(training_loss_arr), 4)}\n\t\t"
                  f"Accuracy: {np.round(np.mean(training_accuracy_arr), 4)}\n\t\t"
                  f"Jaccard: {np.round(np.mean(training_jaccard_arr), 4)}**\n\n")

            print(f"** Validation\n\t\tLoss: {np.round(np.mean(validation_loss_arr), 4)}\n\t\t"
                  f"Accuracy: {np.round(np.mean(validation_accuracy_arr), 4)}\n\t\t"
                  f"Jaccard: {np.round(np.mean(validation_jaccard_arr), 4)}**\n\n")

        if epoch % 10 == 0:
            # Save model each 10 epochs
            save_model(model, epoch, optimizer, training_loss_arr, validation_loss_arr, training_accuracy_arr,
                       validation_accuracy_arr, training_jaccard_arr, validation_jaccard_arr,
                       f"model_epoch_{epoch}.pth")

        early_stopping(validation_loss_arr)
        if early_stopping.early_stop or epoch == num_epochs - 1:
            save_model(model, epoch, optimizer, training_loss_arr, validation_loss_arr, training_accuracy_arr,
                       validation_accuracy_arr, training_jaccard_arr, validation_jaccard_arr, "best_model.pth")

            # Stop training
            break

        lr_scheduler.step()

    return plot_train_loss, plot_validate_loss, plot_validate_accuracy


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
    test_jaccard_arr = []

    # Parameters to plot the results
    plot_img = []
    plot_mask = []
    plot_predicted = []

    # Test
    with torch.no_grad():
        model.eval()

        # Test steps
        test_loss_batch = 0.0
        test_accuracy_batch = 0.0
        test_jaccard_batch = 0.0
        num_steps = 0

        for index, batch in enumerate(test_loader):
            torch.cuda.empty_cache()

            X = batch[0].to(device)
            y = batch[1].to(device)
            y[y > 0] = 1

            y_predicted = model(X)

            plot_img.append(X)
            plot_mask.append(y)
            plot_predicted.append(y_predicted.cpu().detach().numpy())

            loss = loss_fn(y_predicted, y)
            test_loss_batch += loss.item()

            test_accuracy_batch += accuracy(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())
            test_jaccard_batch += jc(y_true=y.cpu().detach().numpy(), y_pred=y_predicted.cpu().detach().numpy())

            num_steps += 1

        test_loss_arr.append(np.divide(test_loss_batch, num_steps))
        test_accuracy_arr.append(np.divide(test_accuracy_batch, num_steps))
        test_jaccard_arr.append(np.divide(test_jaccard_batch, num_steps))

        print(f"** Test\n\t\tLoss: {np.round(np.mean(test_loss_arr), 4)}\n\t\t"
              f"Accuracy: {np.round(np.mean(test_accuracy_arr), 4)}\n\t\t"
              f"Jaccard: {np.round(np.mean(test_jaccard_arr), 4)}**\n\n")

    return plot_img, plot_mask, plot_predicted


def save_model(model, epoch, optimizer, training_loss, validation_loss, training_acc, validation_acc, training_jaccard,
               validation_jaccard, name_file):
    r"""
    Save the model.
    :param model: model to saved.
    :param epoch: current epoch.
    :param optimizer: net optimizer.
    :param training_loss: training loss.
    :param training_acc: training accuracy.
    :param validation_loss: validation loss.
    :param validation_acc: validation accuracy.
    :param training_jaccard: training jaccard index.
    :param validation_jaccard: validation jaccard index.
    :param name_file: name file to be saved. (Extension file .pth)
    """

    current_date_hour = datetime.datetime.now()
    new_dir = f"{current_date_hour.year}{current_date_hour.month}{current_date_hour.day}-" \
              f"{current_date_hour.hour}{current_date_hour.minute}{current_date_hour.second}"

    path = PARENT_DIR + "/" + new_dir
    os.mkdir(path)

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'Training Loss': np.round(np.mean(training_loss, 4)),
        'Validation Loss': np.round(np.mean(validation_loss, 4)),
        'Training Accuracy': np.round(np.mean(training_acc, 4)),
        'Validation Accuracy': np.round(np.mean(validation_acc, 4)),
        'Training Jaccard': np.round(np.mean(training_jaccard, 4)),
        'Validation Jaccard': np.round(np.mean(validation_jaccard, 4))
    }

    torch.save(checkpoint, PARENT_DIR + "/" + new_dir + "/" + name_file)
    print("Model saved!")
