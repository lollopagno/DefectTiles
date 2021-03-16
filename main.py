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

    for img in imgs:
        hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
        plt.plot(bin_edges[0:-1], hist)

    # TODO added labels to histogram
    plt.title("Histogram")
    plt.xlabel("Grayscale values")
    plt.ylabel("Pixels")
    plt.xlim([0.0, 255.0])
    plt.show()


def preprocessing(img):
    r"""
    Performs pre-processing operations
    :param img: image to be processed
    :return: pre-processed image
    """

    # Normalization
    img_norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    histogram([img_gray, img_norm], ["Grayscale", "Normalized"])

    # Median filter (noise reduction)
    img_filtered = cv.medianBlur(img_norm, 3)

    # Median value grayscale
    median_value = img_filtered.mean()

    # Edge Detection
    img_edge = cv.Canny(img_filtered, 0.66 * median_value, 1.33 * median_value)

    return img_edge


def detect_crack(img):
    height, width = img.shape
    for x in range(0, height):
        for y in range(0, width):
            if img[x, y] == 1:
                pass


def detect_pinhole_defects():
    p_count = 0  # Pinhole count
    c_range = 0  # Range del corner
    e_range = 0  # Range degli edge
    row = 0  # Maximum number of image pixels along any row
    col = 0  # Maximum number of image pixels along any column


img = cv.imread("Resources/crackTile.jpg")
cv.imshow("Img original", img)

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

img_pre_processing = preprocessing(img_gray)
# TODO aggiungere conteggio dei pixel neri per essere confrontato con l'immagine di test

cv.imshow("Img result canny", img_pre_processing)
cv.waitKey(0)
