import matplotlib.pyplot as plt
import numpy as np
import random


def plot_history(loss_train, loss_valid, accuracy_valid, IoU_valid, num_epochs):
    r"""
    Shows the results obtained.
    :param loss_train: loss obtained from the training data
    :param loss_valid: loss obtained from the validation data
    :param accuracy_valid: accuracy obtained
    :param IoU_valid: intersection over union obtained
    :param num_epochs: number of epochs
    """

    plt.plot(range(1, num_epochs + 1), loss_train, color='r', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), loss_valid, color='g', label='Validation Loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.title('Loss history')
    plt.show()

    plt.plot(range(1, num_epochs + 1), accuracy_valid, color='r', label='Accuracy')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.title('Accuracy history')
    plt.show()

    plt.plot(range(1, num_epochs + 1), IoU_valid, color='r', label="IoU")
    plt.xlabel('Epoch number')
    plt.ylabel('IoU')
    plt.legend(loc='upper right')
    plt.title('IoU history')
    plt.show()


def sample_dataset(data_loader, batch_size):
    r"""
    Shows an example of the training dataset
    :param data_loader: dataset
    :param batch_size: number of batch size
    """

    counter = 0
    random_seed_1 = random.randint(0, len(data_loader) - 1)
    random_seed_2 = random.randint(0, len(data_loader) - 1)
    for sample_image, sample_label in data_loader:

        if counter == random_seed_1 or counter == random_seed_2:
            random_batch = random.randint(0, batch_size - 1)
            plot_samples(random_batch, sample_image, sample_label)
        counter += 1


def plot_samples(index, img, label):
    r"""
    Plot examples of the dataset.
    :param index: index of the array
    :param img: img to plot
    :param label: label to plot
    """

    sample_image = img[index]
    sample_label = label[index]

    img = np.squeeze(sample_image)
    img = img[0, :, :]
    plt.title('Image')
    plt.imshow(img, cmap="gray")
    plt.show()

    label = np.squeeze(sample_label)
    plt.title('Label')
    plt.imshow(label, cmap="gray")
    plt.show()


def plot_test_results(images, masks, predicted, value):
    r"""
    PLot test results.
    :param images: collection of images of the dataset.
    :param masks: collection of masks of the dataset.
    :param predicted: images predicted by the network.
    :param value: initial value of loop.
    """

    if value >= len(images):
        raise Exception("The value must be less than the number of images")

    counter = 1

    for i in range(value, len(images)):
        for k in range(images[i].shape[0]):
            img = images[i][k]
            img = np.squeeze(img.detach().cpu().numpy())
            img = img[0, :, :]

            mask = masks[i][k]
            mask = mask[0].detach().cpu().numpy()

            predict = predicted[i][k]
            predict = predict[0].detach().cpu().numpy()

            # Image
            plt.title(f'Image: {counter}')
            plt.imshow(img, cmap="gray")
            plt.show()

            # Mask
            plt.title(f'Mask: {counter}')
            plt.imshow(mask, cmap="gray")
            plt.show()

            # Predict
            plt.title(f'Predict: {counter}')
            plt.imshow(predict, cmap="gray")
            plt.show()

            counter += 1

            # TODO delte in the future
            if counter == 2:
                break

        if counter == 2:
            break