import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

CANNY = "Canny"
MEDIAN_BLUR = "Median"
GAUSSIAN_BLUR = "Gaussian"
PATH_IMAGES = "Resources/Histogram/Hist"
CRACKS = "Cracks"


def start(img_original, filter, edge_detection, defect):
    r"""
    Performs pre-processing operations
    :param img_original: image to be processed
    :param filter: type of filter to apply
    :param edge_detection: edge detection method (canny, sobel)
    :param defect: type of defect to be identified
    :return: pre-processed image to detect defects
    """

    # Conversion color from RGB to grayscale
    img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    # Normalization
    img_norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    if defect == CRACKS:
        # Blurring
        img_norm = cv.blur(img_norm, (3, 3))

        # Gamma correction
        img_norm = correction_gamma(img_original, img_norm, gamma=2.0)

    # Applying the filter (noise reduction)
    if filter == MEDIAN_BLUR:  # Median
        img_filt = cv.medianBlur(img_norm, 3)
    elif filter == GAUSSIAN_BLUR:  # Gaussian
        img_filt = cv.GaussianBlur(img_norm, (3, 3), 0)
    else:  # Bilateral filter
        img_filt = cv.bilateralFilter(img_norm, 3, 75, 75)

    # Edge Detection
    if edge_detection == CANNY:
        img_edge = cv.Canny(img_filt, 50, 150)

    else:

        scale = 1
        delta = 0
        ddepth = cv.CV_16S

        grad_x = cv.Sobel(img_filt, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img_filt, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        img_edge = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Closing
    kernel = np.ones((5, 5), np.uint8)
    img_closing = cv.morphologyEx(img_edge, cv.MORPH_CLOSE, kernel)

    result_edge = thresholding_image(img_filt, img_closing)
    return result_edge


def histogram(img, file_name):
    r"""
    Create the histogram and save it.
    :param img: image of histogram to be created
    :param file_name: file name to save it
    """

    plt.title("Histogram")
    plt.xlabel("Grayscale values")
    plt.ylabel("Pixels")
    plt.xlim([0.0, 255.0])

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    plt.plot(bin_edges[0:-1], hist)
    plt.savefig(PATH_IMAGES + file_name)
    plt.clf()


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


def correction_gamma(img_original, img, gamma=0.50):
    r"""
    Apply gamma correction
    :param img_original: image in which to calculate the average
    :param img: image in which to apply gamma correction
    :param gamma: gamma value to apply
    :return: resulting image of the correction
    """

    mean = cv.mean(img_original)
    if mean[0] >= 100 and mean[0] == mean[1] == mean[2]:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)

        img = cv.LUT(img, table)

    return img
