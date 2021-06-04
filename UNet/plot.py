import matplotlib.pyplot as plt
import numpy as np
import random


def plot_history(loss_train, loss_valid, accuracy_valid, num_epochs):
    r"""
    Shows the results obtained.
    :param loss_train: loss obtained from the training data
    :param loss_valid: loss obtained from the validation data
    :param accuracy_valid: accuracy obtained
    :param num_epochs: number of epochs
    """

    plot_epoch = np.arange(0, num_epochs)

    plt.figure(1)
    plt.plot(plot_epoch[0:num_epochs], loss_train[0:num_epochs], linestyle='-', marker='',
             linewidth=3, alpha=0.9)
    plt.plot(plot_epoch[0:num_epochs], loss_valid[0:num_epochs], linestyle='-', marker='',
             linewidth=3, alpha=0.9)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.gca().legend(('Training loss', 'Validation loss'), loc='upper right')
    plt.title('Loss history')
    plt.show()

    plt.figure(2)
    plt.plot(plot_epoch[0:num_epochs], accuracy_valid[0:num_epochs], linestyle='-', marker='',
             linewidth=3, alpha=0.9)
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.gca().legend('Accuracy', loc='upper right')
    plt.title('Accuracy history')
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

    for i in range(value, len(images)):
        img = images[i]
        mask = masks[i]
        predict = predicted[i]

        # Image
        plt.title('Image')
        plt.imshow(img, cmap="gray")
        plt.show()

        # Label
        plt.title('Mask')
        plt.imshow(mask, cmap="gray")
        plt.show()

        # Predict
        plt.title('Predict')
        plt.imshow(predict, cmap="gray")
        plt.show()
