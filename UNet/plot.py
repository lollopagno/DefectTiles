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
    plt.plot(plot_epoch[0:num_epochs], loss_train[0:num_epochs], linestyle='--', marker='',
             linewidth=3, alpha=0.9)
    plt.plot(plot_epoch[0:num_epochs], loss_valid[0:num_epochs], linestyle='-', marker='',
             linewidth=3, alpha=0.9)
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.gca().legend(('training loss', 'validation loss'), loc='upper right')
    plt.title('losses against epoch number')

    plt.figure(2)
    plt.plot(plot_epoch[0:num_epochs], accuracy_valid[0:num_epochs], linestyle='-', marker='',
             linewidth=3, alpha=0.9)
    plt.xlabel('epoch number')
    plt.ylabel('accuracy')
    plt.gca().legend('accuracy', loc='lower right')
    plt.title('validation accuracy against epoch number')


def sample_dataset(dataLoader, batch_size):
    r"""
    Shows an example of the training dataset
    :param dataLoader: dataset
    :param batch_size: number of batchsize
    """

    # TODO Improved this, if necessary
    for test_images, test_labels in dataLoader:
        random_batch = random.randint(0, batch_size - 1)

        sample_image = test_images[random_batch]
        sample_label = test_labels[random_batch]

        img = np.squeeze(sample_image)
        plt.title('Image')
        plt.imshow(img)
        plt.show()

        label = np.squeeze(sample_label)
        plt.title('Label')
        plt.imshow(label)
        plt.show()
        break
