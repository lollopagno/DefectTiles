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


def start(img_original, filter, edge_detection):
    r"""
    Performs pre-processing operations
    :param img_original: image to be processed
    :param filter: type of filter to apply
    :param edge_detection: edge detection method (canny, sobel)
    :return: pre-processed image
    """

    # Conversion color from RGB to grayscale
    img = cv.cvtColor(img_original, cv.COLOR_RGB2GRAY)

    # Normalization
    img_norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Applying the filter (noise reduction)
    if filter == "Median":
        img_filt = cv.medianBlur(img_norm, 3)
    elif filter == "Gaussian":
        img_filt = cv.GaussianBlur(img_norm, (3, 3), 0)
    else:
        img_filt = cv.bilateralFilter(img_norm, 3, 75, 75)
    # histogram([img, img_filt], ["Grayscale", "Filtered"])

    # Edge Detection
    if edge_detection == "Canny":
        # median_value = img_filt.mean()

        # img_edge = cv.Canny(img_filt, 0.66 * median_value, 1.33 * median_value)  #TODO valutare se considerare il valore medio
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

    img_edge = improved_image(img_original, img_edge)

    return img_edge


def improved_image(img, img_edge):
    r"""
    Performs a threshold operation by deleting components specified by threshold values
    :param img: original image
    :param img_edge: binary image that contains the edges
    :return: binary image in which to find for defects
    """

    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h_min, s_min, v_min = 0, 0, 107
    h_max, s_max, v_max = 0, 0, 255

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(imgHSV, lower, upper)
    img_result = cv.bitwise_and(img, img, mask=mask)
    img_result = cv.dilate(img_result, (3, 3), iterations=3)
    img_result = cv.cvtColor(img_result, cv.COLOR_BGR2GRAY)

    subtract = cv.subtract(img_edge, img_result)

    return subtract
