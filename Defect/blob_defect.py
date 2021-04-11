import numpy as np
import cv2 as cv


def detect(original, img, method="Sobel"):
    if method == "Sobel":
        _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img = cv.dilate(img, element, iterations=2)
    img = cv.erode(img, element, iterations=1)

    cv.imshow("After morphologic", img)
    contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

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
        cv.drawContours(original, cnt, -1, (0, 0, 255), 3)

    return original