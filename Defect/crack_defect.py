import numpy as np
import cv2 as cv
from Defect import common as utility

SOBEL = "Sobel"
CRACKS = "Cracks"
RED = np.array([0, 0, 255])
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def detect(img_original, img_edge, method=SOBEL):
    r"""
    Detects cracks in the image
    :param img_original: original image in which to draw the defects
    :param img_edge: binary image that contains the edges
    :param method: edge detection method (canny, sobel)
    :return: original image with cracks detected, binary image with cracks detected
    """

    cracks = connected_components(img_edge / 255, method)
    cracks_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)

    mean_original_img = cv.mean(img_original)

    if len(cracks) != 0:
        for crack in cracks:

            intensity_pixels = []
            contours = []
            for _ in range(0, len(crack)):
                x, y = crack.pop()
                contours.append([y, x])
                intensity_pixels.append(img_original[x, y][0])

            mean_area = np.mean(intensity_pixels)

            if mean_original_img[0] == mean_original_img[1]:
                if mean_area < mean_original_img[0]:
                    contours = np.array(contours).astype(np.int32)
                    cv.drawContours(cracks_detect, [contours], -1, WHITE, -1)
            else:
                contours = np.array(contours).astype(np.int32)
                cv.drawContours(cracks_detect, [contours], -1, WHITE, -1)

        # Find for the contours of the identified cracks
        contours, _ = cv.findContours(cracks_detect.copy().astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cracks_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)

        for cnt in contours:
            if utility.calc_distance(cnt, CRACKS):
                cv.drawContours(cracks_detect, [cnt], -1, WHITE, -1)
                cv.polylines(img_original, cnt, -1, GREEN, 2)

    return img_original, cracks_detect.astype(np.uint8)


def connected_components(img, method):
    r"""
    Detect connected components in an image
    :param method: edge detection method (canny, sobel)
    :param img: image in which to detect connected components
    :return: stack with the coordinates of the detected cracks
    """

    height, width = img.shape

    if method == SOBEL:
        value_found = 0.098
        crack_lenght = 200

    else:
        value_found = 1
        crack_lenght = 20

    visited = np.zeros((height, width), dtype=bool)

    tmp_stack = []
    coordinates_result_cracks = []
    coordinates_current_component = []

    # Depth-first search (DFS)
    for i in range(0, height):
        for j in range(0, width):
            lenght_components = 0
            height_components = 0
            width_components = 0

            if visited[i, j]:  # If i have already visited it, continue
                continue

            elif img[i, j] == 0:
                visited[i, j] = True  # I mark it as visited

            else:
                visited[i, j] = True
                tmp_stack.append((i, j))

                while len(tmp_stack) != 0:

                    x, y = tmp_stack.pop()

                    lenght_components += 1
                    coordinates_current_component.append((x, y))

                    if x - 1 >= 0 and y - 1 >= 0:
                        p1 = img[x - 1, y - 1]
                        if p1 >= value_found and not visited[x - 1, y - 1]:
                            tmp_stack.append((x - 1, y - 1))
                            visited[x - 1, y - 1] = True

                            height_components += 0.5
                            width_components += 0.5

                    if x - 1 >= 0:
                        p2 = img[x - 1, y]
                        if p2 >= value_found and not visited[x - 1, y]:
                            tmp_stack.append((x - 1, y))
                            visited[x - 1, y] = True

                            height_components += 1

                    if x - 1 >= 0 and y + 1 < width:
                        p3 = img[x - 1, y + 1]
                        if p3 >= value_found and not visited[x - 1, y + 1]:
                            tmp_stack.append((x - 1, y + 1))
                            visited[x - 1, y + 1] = True

                            height_components += 0.5
                            width_components += 0.5

                    if y - 1 >= 0:
                        p4 = img[x, y - 1]
                        if p4 >= value_found and not visited[x, y - 1]:
                            tmp_stack.append((x, y - 1))
                            visited[x, y - 1] = True

                            width_components += 1

                    if y + 1 < width:
                        p5 = img[x, y + 1]
                        if p5 >= value_found and not visited[x, y + 1]:
                            tmp_stack.append((x, y + 1))
                            visited[x, y + 1] = True

                            width_components += 1

                    if x + 1 < height and y - 1 >= 0:
                        p6 = img[x + 1, y - 1]
                        if p6 >= value_found and not visited[x + 1, y - 1]:
                            tmp_stack.append((x + 1, y - 1))
                            visited[x + 1, y - 1] = True

                            height_components += 0.5
                            width_components += 0.5

                    if x + 1 < height:
                        p7 = img[x + 1, y]
                        if p7 >= value_found and not visited[x + 1, y]:
                            tmp_stack.append((x + 1, y))
                            visited[x + 1, y] = True

                            height_components += 1

                    if x + 1 < height and y + 1 < width:
                        p8 = img[x + 1, y + 1]
                        if p8 >= value_found and not visited[x + 1, y + 1]:
                            tmp_stack.append((x + 1, y + 1))
                            visited[x + 1, y + 1] = True

                            height_components += 0.5
                            width_components += 0.5

                if lenght_components >= crack_lenght:

                    max_value = max((height_components, width_components))
                    min_value = min((height_components, width_components))
                    diff = max_value - min_value

                    if diff >= 15.0:
                        # Crack cetected
                        coordinates_result_cracks.append(coordinates_current_component.copy())

                coordinates_current_component.clear()

    return coordinates_result_cracks
