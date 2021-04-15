import numpy as np
import cv2 as cv


def detect(img_original, img_edge, method="Sobel"):
    r"""
    Detects blobs in the image
    :param img_original: original image in which to draw the defects
    :param img_edge: image in which to detect blobs
    :param method: edge detection method (canny, sobel)
    :return:
    """
    if method == "Sobel":
        _, img_edge = cv.threshold(img_edge, 50, 255, cv.THRESH_BINARY)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img_edge = cv.dilate(img_edge, element, iterations=2)
    img_edge = cv.erode(img_edge, element, iterations=1)

    # cv.imshow("After morphologic", img)
    contours, hierarchy = cv.findContours(img_edge, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    centers = []
    contours_to_draw = []
    for cnt in contours:

        area = cv.contourArea(cnt)

        if area < 10:
            continue

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.05 * peri, True)
        objCor = len(approx)
        if objCor > 4:
            contours_to_draw.append(cnt)
            m = cv.moments(cnt)
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            centers.append(center)

    for cnt in contours_to_draw:
        # cv.circle(original, center, 3, (255, 0, 0), -1)
        cv.drawContours(img_original, cnt, -1, (0, 0, 255), 3)

    return img_original
