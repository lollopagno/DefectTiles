import cv2 as cv
import numpy as np
import math

CRACKS = "Cracks"
MIN_DISTANCE_CRACK = 5
RANGE_MIN_RADIUS_CRACK = 10


def calculate_circolarity(contour, area, defect):
    r"""
    Calculate ....
    :param area:
    :param contour: current contour
    :param defect: type of defect
    :return: true if the all distances is greater or less than the minimum distance
    """

    perimeter = cv.arcLength(contour, -1)
    circularity = (4 * math.pi * area) / math.pow(perimeter, 2)

    if defect == CRACKS:
        return circularity <= 0.35

    else:

        if circularity >= 0.8:
            # Circle detected
            return True

        elif circularity >= 0.5:

            # Potential ellipse
            ellipse = cv.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = round(np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2), 2)
            return eccentricity >= 0.8


def connected_components(img):
    r"""
    Detect connected components in an image
    :param img: image in which to detect connected components
    :return: stack with the coordinates of the detected cracks
    """

    height, width = img.shape

    value_found = 1
    defect_lenght = 20

    visited = np.zeros((height, width), dtype=bool)

    tmp_stack = []
    coordinates_result_defect = []
    coordinates_current_component = []

    # Depth-first search (DFS)
    for i in range(0, height):
        for j in range(0, width):
            lenght_components = 0

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

                    if x - 1 >= 0:
                        p2 = img[x - 1, y]
                        if p2 >= value_found and not visited[x - 1, y]:
                            tmp_stack.append((x - 1, y))
                            visited[x - 1, y] = True

                    if x - 1 >= 0 and y + 1 < width:
                        p3 = img[x - 1, y + 1]
                        if p3 >= value_found and not visited[x - 1, y + 1]:
                            tmp_stack.append((x - 1, y + 1))
                            visited[x - 1, y + 1] = True

                    if y - 1 >= 0:
                        p4 = img[x, y - 1]
                        if p4 >= value_found and not visited[x, y - 1]:
                            tmp_stack.append((x, y - 1))
                            visited[x, y - 1] = True

                    if y + 1 < width:
                        p5 = img[x, y + 1]
                        if p5 >= value_found and not visited[x, y + 1]:
                            tmp_stack.append((x, y + 1))
                            visited[x, y + 1] = True

                    if x + 1 < height and y - 1 >= 0:
                        p6 = img[x + 1, y - 1]
                        if p6 >= value_found and not visited[x + 1, y - 1]:
                            tmp_stack.append((x + 1, y - 1))
                            visited[x + 1, y - 1] = True

                    if x + 1 < height:
                        p7 = img[x + 1, y]
                        if p7 >= value_found and not visited[x + 1, y]:
                            tmp_stack.append((x + 1, y))
                            visited[x + 1, y] = True

                    if x + 1 < height and y + 1 < width:
                        p8 = img[x + 1, y + 1]
                        if p8 >= value_found and not visited[x + 1, y + 1]:
                            tmp_stack.append((x + 1, y + 1))
                            visited[x + 1, y + 1] = True

                if lenght_components >= defect_lenght:
                    # Degect cetected
                    coordinates_result_defect.append(coordinates_current_component.copy())

                coordinates_current_component.clear()

    return coordinates_result_defect
