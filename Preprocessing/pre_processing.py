import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def histogram(imgs, labels):
    r"""
    Create N histograms
    :param imgs: number of images of histograms to be created
    :param labels: number of labes of histograms to be created
    """

    if len(imgs) != len(labels):
        raise Exception("The size of images and labels must be the same")

    count_label = 0
    for img in imgs:
        hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
        plt.plot(bin_edges[0:-1], hist, label=labels[count_label])
        plt.legend(loc="upper left")
        count_label += 1

    plt.title("Histogram")
    plt.xlabel("Grayscale values")
    plt.ylabel("Pixels")
    plt.xlim([0.0, 255.0])
    plt.show()


def start(img):
    r"""
    Performs pre-processing operations
    :param img: image to be processed
    :return: pre-processed image
    """

    # Normalization
    img_norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Median filter (noise reduction)
    img_filt = cv.medianBlur(img_norm, 3)
    # histogram([img, img_filt], ["Grayscale", "Filtered"])

    # Edge Detection
    median_value = img_filt.mean()
    img_edge = cv.Canny(img_filt, 0.66 * median_value, 1.33 * median_value)

    # TODO aggiungere conteggio dei pixel neri per essere confrontato con l'immagine di test

    return img_edge / 255.0