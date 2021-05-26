import numpy as np
import cv2 as cv
from Features.Defect import common as utility

BLOBS = "Blob"
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


def detect(img_original, img_edge):
    r"""
    Detects blobs in the image
    :param img_original: original image in which to draw the defects
    :param img_edge: image in which to detect blobs
    :return: original orignal image with blobs detected
    """

    blobs = utility.connected_components(img_edge / 255)
    img_blobs_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)

    mean_original_img = round(cv.mean(img_original)[0])

    if len(blobs) != 0:
        for blob in blobs:

            intensity_pixels = []
            contours_blob = []

            for _ in range(0, len(blob)):
                x, y = blob.pop()
                contours_blob.append([y, x])
                intensity_pixels.append(img_original[x, y][0])

            avg_luminance_area_contour = round(float(np.mean(intensity_pixels)))

            if avg_luminance_area_contour <= mean_original_img:
                contours_blob = np.array(contours_blob).astype(np.int32)
                cv.drawContours(img_blobs_detect, [contours_blob], -1, WHITE, -1)

        # Kernel
        kernel_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        dilate = cv.dilate(img_blobs_detect, kernel_3)
        img_closing = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel_5)
        contours_blob, _ = cv.findContours(img_closing.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        img_blobs_detect = np.zeros((img_original.shape[:2]), dtype=np.uint8)

        # Detect blobs
        for blob in contours_blob:

            area = cv.contourArea(blob)
            if area < 20:
                continue

            peri = cv.arcLength(blob, -1)  # Perimeter
            approx = cv.approxPolyDP(blob, 0.05 * peri, -1)
            objCor = len(approx)

            if objCor >= 4:

                if utility.calc_geometric_descriptors(blob, area, BLOBS):
                    cv.drawContours(img_blobs_detect, [blob], -1, WHITE, -1)
                    cv.polylines(img_original, [blob], -1, RED, 2)

    return img_original
