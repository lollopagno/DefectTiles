import numpy as np
import cv2 as cv
from Defect import common as utility

RED = np.array([0, 0, 255])
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

CRACKS = "Cracks"


def detect(img_original, img_edge):
    r"""
    Detects cracks in the image
    :param img_original: original image in which to draw the defects
    :param img_edge: binary image that contains the edges
    :return: original image with cracks detected, binary image with cracks detected
    """

    cracks = utility.connected_components(img_edge / 255)
    cracks_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)

    mean_original_img = cv.mean(img_original)[0]

    if len(cracks) != 0:
        for crack in cracks:

            intensity_pixels = []
            contours_crack = []
            for _ in range(0, len(crack)):
                x, y = crack.pop()
                contours_crack.append([y, x])
                intensity_pixels.append(img_original[x, y][0])

            avg_luminance_area_contour = np.mean(intensity_pixels)

            if avg_luminance_area_contour < mean_original_img:
                contours_crack = np.array(contours_crack).astype(np.int32)
                cv.drawContours(cracks_detect, [contours_crack], -1, WHITE, -1)

        # Find for the contours of the identified cracks
        contours_crack, _ = cv.findContours(cracks_detect.copy().astype(np.uint8), cv.RETR_EXTERNAL,
                                            cv.CHAIN_APPROX_NONE)
        cracks_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)

        for cnt in contours_crack:

            area = cv.contourArea(cnt)
            if area < 20:
                continue

            if utility.calculate_circolarity(cnt, area, CRACKS):
                cv.drawContours(cracks_detect, [cnt], -1, WHITE, -1)
                cv.polylines(img_original, cnt, -1, GREEN, 2)

    return img_original, cracks_detect.astype(np.uint8)
