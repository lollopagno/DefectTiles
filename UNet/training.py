import torch
import numpy as np
from tqdm import tqdm
from UNet.metric import accuracy


def training_loop(model, num_epochs, optimizer, lr_scheduler, loss_fn, training_loader, validation_loader):
    r"""
    # TODO documentation
    :param model
    :param num_epochs
    :param optimizer
    :param lr_scheduler
    :param loss_fn
    :param training_loader
    :param validation_loader
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialized params before training
    training_loss_arr = []
    validation_loss_arr = []

    training_accuracy_arr = []
    validation_accuracy_arr = []

    epoch_count = 0

    for _ in tqdm(range(0, num_epochs)):

        lr_scheduler.step()

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
            y = batch[1].to(device)  # TODO batch_label[batch_label>0] = 1

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

        training_loss_arr.append(np.divide(training_loss_batch, num_steps))
        training_accuracy_arr.append(np.divide(training_accuracy_batch, num_steps))

        # TODO check print
        print(f"** Training\nEpoch: {epoch_count + 1}\nLoss: {np.mean(training_loss_arr)}\n"
              f"Accuracy: {np.mean(training_accuracy_arr)}\n**\n\n")

        # Validation steps
        with torch.no_grad():

            model.eval()

            validation_loss_batch = 0.0
            validation_accuracy_batch = 0.0
            num_steps = 0

            for index, batch in enumerate(validation_loader):
                torch.cuda.empty_cache()

                X = batch[0].to(device)
                y = batch[1].to(device)  # TODO batch_label[batch_label>0] = 1

                y_predicted = model(X)
                loss = loss_fn(y_predicted, y)
                validation_loss_batch += loss.item()
                validation_accuracy_batch += accuracy(y_predicted.cpu().detach().numpy(),
                                                      y.cpu().detach().numpy())

                num_steps += 1

            validation_loss_arr.append(np.divide(validation_loss_batch, num_steps))
            validation_accuracy_arr.append(np.divide(validation_accuracy_batch, num_steps))
            print(f"** Validation\nEpoch: {epoch_count + 1}\nLoss: {np.mean(validation_loss_arr)}\n"
                  f"Accuracy: {np.mean(validation_accuracy_arr)}\n**\n\n")

        epoch_count += 1

    # TODO added training test

    return training_loss_arr, validation_loss_arr, training_accuracy_arr, validation_accuracy_arr
