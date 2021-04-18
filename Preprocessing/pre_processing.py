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


def start(img_original, filter, edge_detection, blurring=True):
    r"""
    Performs pre-processing operations
    :param img_original: image to be processed
    :param filter: type of filter to apply
    :param edge_detection: edge detection method (canny, sobel)
    :param blurring: whether to apply blurring
    :return: pre-processed image to detect defects
    """

    # Conversion color from RGB to grayscale
    img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    # Normalization
    img_norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Blurring
    if blurring:
        img_norm = cv.blur(img_norm, (3, 3))

    # Applying the filter (noise reduction)
    if filter == "Median":  # Median
        img_filt = cv.medianBlur(img_norm, 3)
    elif filter == "Gaussian":  # Gaussian
        img_filt = cv.GaussianBlur(img_norm, (3, 3), 0)
    else:  # Bilateral filter
        img_filt = cv.bilateralFilter(img_norm, 3, 75, 75)
    # histogram([img, img_filt], ["Grayscale", "Filtered"])

    # Edge Detection
    if edge_detection == "Canny":
        img_edge = cv.Canny(img_filt, 50, 150)

    elif edge_detection == "Sobel":

        scale = 1
        delta = 0
        ddepth = cv.CV_16S

        grad_x = cv.Sobel(img_filt, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img_filt, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        img_edge = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    else:
        raise Exception("Specify the edge detection method: Canny or Sobel")

    # Closing
    kernel = np.ones((5, 5), np.uint8)
    img_closing = cv.morphologyEx(img_edge, cv.MORPH_CLOSE, kernel)

    result_edge = thresholding_image(img_filt, img_closing)
    return result_edge


def thresholding_image(img, img_edge):
    r"""
    Apply binarization of otsu followed by morphological and bit operations
    :param img: original image
    :param img_edge: binary image that contains the edges
    :return: binary image in which to find for defects
    """

    # Otsu threshold
    _, otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Bit operation
    otsu_not = cv.bitwise_not(otsu)
    img_and = cv.bitwise_and(otsu_not, img_edge)

    # Closing
    kernel = np.ones((7, 7), np.uint8)
    img_closing = cv.morphologyEx(img_and, cv.MORPH_CLOSE, kernel)

    return img_closing
