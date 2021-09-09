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

    plt.plot(range(1, num_epochs), loss_train, color='r', label='Training Loss')
    plt.plot(range(1, num_epochs), loss_valid, color='g', label='Validation Loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.title('Loss history')
    plt.show()

    plt.plot(range(1, num_epochs), accuracy_valid, color='r', label='Accuracy')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.title('Accuracy history')
    plt.show()

    plt.plot(range(1, num_epochs), IoU_valid, color='r', label="IoU")
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


def plot_test_results(images, masks, predicted, rows=4):
    r"""
    PLot test results.
    :param images: collection of images of the dataset.
    :param masks: collection of masks of the dataset.
    :param predicted: images predicted by the network.
    :param rows: row table.
    """

    if rows >= len(images):
        raise Exception("The rows table must be less than the number of images!")

    images_copy = images.copy()
    masks_copy = masks.copy()
    predicted_copy = predicted.copy()

    fig, axs = plt.subplots(rows, 3, figsize=(32, 16))
    axs = axs.ravel()

    counter = 0

    for i in range(rows):
        index = random.randint(0, len(images_copy) - 1)
        img = images_copy[index]
        mask = masks_copy[index]
        predict = predicted_copy[index]

        img = img.permute(1, 2, 0).cpu().numpy()
        mask = mask.permute(1, 2, 0).cpu().numpy()
        predict = predict.permute(1, 2, 0).cpu().numpy()

        del images_copy[index]
        del masks_copy[index]
        del predicted_copy[index]

        axs[i + counter].imshow(img)
        axs[i + 1 + counter].imshow(mask)
        axs[i + 2 + counter].imshow(predict)

        counter += 2

    plt.show()
