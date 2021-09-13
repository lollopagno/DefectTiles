import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

PATH_IMAGES = "Features/Resources/Histogram/Hist"
CRACKS = "Cracks"
MIN_DISTANCE_CRACK = 5
RANGE_MIN_RADIUS_CRACK = 10


def calc_geometric_descriptors(contour, area, defect):
    r"""
    Calculation of geometric descriptors for the given contour: circularity, eccentricity
    :param area: contour area
    :param contour: current contour
    :param defect: type of defect
    :return: true if the defect is found otherwise false
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
    Detect connected components in an binary image by searching by depth (DFS).
    :param img: image in which to detect connected components
    :return: stack with the coordinates of the detected defect
    """

    height, width = get_shape(img)

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

            if visited[i, j]:
                # If i have already visited it, continue
                continue

            elif img[i, j] == 0:
                # I mark it as visited
                visited[i, j] = True

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
                    # Defect detected
                    coordinates_result_defect.append(coordinates_current_component.copy())

                coordinates_current_component.clear()

    return coordinates_result_defect

def histogram(img, file_name):
    r"""
    Create the histogram and save it.
    :param img: image of histogram to be created
    :param file_name: file name to save it
    """

    plt.title("Histogram")
    plt.xlabel("Grayscale values")
    plt.ylabel("Pixels")
    plt.xlim([0.0, 255.0])

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    plt.plot(bin_edges[0:-1], hist)
    plt.savefig(PATH_IMAGES + file_name)
    plt.clf()


def get_shape(img):
    r"""
    Gets the shape of the image based on the number of channels
    :param img: image to get shape
    :return: height, width of the image
    """

    try:
        height, width, _ = img.shape
    except:
        height, width = img.shape

    return height, width
