import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_edge = []


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


def start(img, filter, method_edge_detection):
    r"""
    Performs pre-processing operations
    :param filter: type of filter to apply
    :param method_edge_detection: edge detection method (canny, sobel)
    :param img: image to be processed
    :return: pre-processed image
    """

    global img_edge

    # Conversion color from RGB to grayscale
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

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
    if method_edge_detection == "Canny":
        # median_value = img_filt.mean()
        # img_edge = cv.Canny(img_filt, 0.66 * median_value, 1.33 * median_value)  #TODO valutare se considerare il valore medio
        img_edge = cv.Canny(img_filt, 50, 150)

    elif method_edge_detection == "Sobel":

        scale = 1
        delta = 0
        ddepth = cv.CV_16S

        grad_x = cv.Sobel(img_filt, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(img_filt, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        img_edge = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # TODO aggiungere conteggio dei pixel neri per essere confrontato con l'immagine di test
    return img_edge / 255.0
