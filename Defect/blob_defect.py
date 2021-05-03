import numpy as np
import cv2 as cv
from Defect import common as utility

SOBEL = "Sobel"
BLOBS = "Blob"
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


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

    # Kernel
    kernel5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    img_dilate = cv.dilate(img_edge, kernel3)
    img_closing = cv.morphologyEx(img_dilate, cv.MORPH_CLOSE, kernel5)

    contours_blob, _ = cv.findContours(img_closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blob_detect = np.zeros((img_original.shape[:2]), dtype=np.uint8)

    # Detect blobs
    for blob in contours_blob:

        area = cv.contourArea(blob)
        if area < 20:
            continue

        peri = cv.arcLength(blob, -1)  # Perimeter
        approx = cv.approxPolyDP(blob, 0.05 * peri, -1)
        objCor = len(approx)

        if objCor >= 4:

            if utility.calc_distance(blob, BLOBS):
                cv.drawContours(blob_detect, [blob], -1, WHITE, -1)
                cv.polylines(img_original, [blob], -1, RED, 2)

    return img_original
