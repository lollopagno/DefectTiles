import numpy as np
import cv2 as cv
from Defect import common as utility

SOBEL = "Sobel"
BLOBS = "Blob"


def detect(img_original, img_edge, method=SOBEL):
    r"""
    Detects blobs in the image
    :param img_original: original image in which to draw the defects
    :param img_edge: image in which to detect blobs
    :param method: edge detection method (canny, sobel)
    :return:
    """

    if method == SOBEL:
        _, img_edge = cv.threshold(img_edge, 50, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 35), np.uint8)
    img_closing = cv.morphologyEx(img_edge, cv.MORPH_CLOSE, kernel)
    # cv.imshow("Closing", img_closing)

    # element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # img_edge = cv.dilate(img_edge, element, iterations=2)
    # img_ellipse = cv.erode(img_edge, element, iterations=1)
    # cv.imshow("Ellipse", img_ellipse)

    contours, _ = cv.findContours(img_closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv.contourArea(cnt)
        if area < 2.0:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.05 * peri, True)
        objCor = len(approx)

        if 4 <= objCor <= 7:  # TODO cehck this!

            if utility.calc_distance(cnt, BLOBS):
                cv.drawContours(img_original, cnt, -1, (0, 0, 255), 3)

    return img_original
