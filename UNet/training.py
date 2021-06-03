import torch
import numpy as np
from UNet.metric import accuracy
from UNet.earlyStopping import EarlyStopping
import datetime
import os

SHOW_EVERY = 1
PARENT_DIR = "UNet/ModelSaved"


def training_loop(model, num_epochs, optimizer, lr_scheduler, loss_fn, training_loader, validation_loader, test_loader):
    r"""
    Network training
    :param model: neural network model
    :param num_epochs: number of epochs
    :param optimizer: optimize used
    :param lr_scheduler: scheduler used
    :param loss_fn: loss function used
    :param training_loader: training data
    :param validation_loader: validation data
    :param test_loader: test data
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialized params before training
    training_loss_arr = []
    validation_loss_arr = []
    test_loss_arr = []

    training_accuracy_arr = []
    validation_accuracy_arr = []
    test_accuracy_arr = []

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
        num_steps = 0

        for _, batch in enumerate(training_loader):
            torch.cuda.empty_cache()

            # Training model
            model.train()

            # Split the train data from the labels
            X = batch[0].to(device)
            y = batch[1].to(device)

            y[y > 0] = 1  # TODO check this

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

            num_steps += 1

        train_loss_for_this_epoch = np.divide(training_loss_batch, num_steps)
        training_loss_arr.append(train_loss_for_this_epoch)
        training_accuracy_arr.append(np.divide(training_accuracy_batch, num_steps))

        # Set loss (training) for each epoch
        plot_train_loss[epoch] = train_loss_for_this_epoch

        # Validation
        with torch.no_grad():

            model.eval()

            # Validation steps
            validation_loss_batch, validation_accuracy_batch, num_steps = _training_loop(validation_loader,
                                                                                         model,
                                                                                         loss_fn,
                                                                                         device)

            valid_loss_for_this_epoch = np.divide(validation_loss_batch, num_steps)
            valid_accuracy_for_this_epoch = np.divide(validation_accuracy_batch, num_steps)

            validation_loss_arr.append(valid_loss_for_this_epoch)
            validation_accuracy_arr.append(valid_accuracy_for_this_epoch)

            # Set loss and accuracy (validation) for each epoch
            plot_validate_loss[epoch] = train_loss_for_this_epoch
            plot_validate_accuracy[epoch] = valid_accuracy_for_this_epoch

        if epoch % SHOW_EVERY == 0 or epoch == num_epochs - 1:
            print(f"** Training\n\t\tLoss: {np.round(np.mean(validation_loss_arr), 4)}\n\t\t"
                  f"Accuracy: {np.round(np.mean(validation_accuracy_arr), 4)}\n**\n\n")

            print(f"** Validation\n\t\tLoss: {np.round(np.mean(training_loss_arr), 4)}\n\t\t"
                  f"Accuracy: {np.round(np.mean(training_accuracy_arr), 4)}\n**\n\n")

        if epoch % 10 == 0:
            # Save model each 10 epochs
            save_model(model, f"model_epoch_{epoch}.pth")

        early_stopping(validation_loss_arr)
        if early_stopping.early_stop or epoch == num_epochs - 1:
            save_model(model, "best_model.pth")

            # Stop training
            break

        lr_scheduler.step()

    # Test
    with torch.no_grad():

        model.eval()

        # Test steps
        test_loss_batch, test_accuracy_batch, num_steps = _training_loop(test_loader,
                                                                         model,
                                                                         loss_fn,
                                                                         device)

        test_loss_arr.append(np.divide(test_loss_batch, num_steps))
        test_accuracy_arr.append(np.divide(test_accuracy_batch, num_steps))
        print(f"** Test\n\t\tLoss: {np.round(np.mean(test_loss_arr), 4)}\n\t\t"
              f"Accuracy: {np.round(np.mean(test_accuracy_arr), 4)}\n**\n\n")

    return plot_train_loss, plot_validate_loss, plot_validate_accuracy


def _training_loop(data_loader, model, loss_fn, device):
    r"""
    :param data_loader: data to be input to the network
    :param model: neural network model
    :param loss_fn: loss function used
    :param: cpu or gpu
    """

    total_loss = 0.0
    total_accuracy = 0.0
    num_steps = 0

    for index, batch in enumerate(data_loader):
        torch.cuda.empty_cache()

        X = batch[0].to(device)
        y = batch[1].to(device)
        y[y > 0] = 1  # TODO check this

        y_predicted = model(X)
        loss = loss_fn(y_predicted, y)
        total_loss += loss.item()

        total_accuracy += accuracy(y_predicted.cpu().detach().numpy(), y.cpu().detach().numpy())

        num_steps += 1

    return total_loss, total_accuracy, num_steps


def save_model(model, name_file):
    r"""
    Save the model.
    :param model: model to saved.
    :param name_file: name file to be saved. (Extension file .pth)
    """

    current_date_hour = datetime.datetime.now()
    new_dir = f"{current_date_hour.year}{current_date_hour.month}{current_date_hour.day}-" \
              f"{current_date_hour.hour}{current_date_hour.minute}{current_date_hour.second}"

    path = PARENT_DIR + "/" + new_dir
    os.mkdir(path)

    torch.save(model.state_dict(), PARENT_DIR + "/" + new_dir + "/" + name_file)
    print("Model saved!")
