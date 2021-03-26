import numpy as np
import cv2 as cv
import sys
from scipy.ndimage.measurements import label

# TODO cercare di far filtrare i crack per:
#  1- forma (linea, scartare i cerchi);
#  2- range delle componenti connesse;
#  3- energia.

def detect(img, method= "Sobel"):
    r"""
    Detects cracks in the image
    :param method: edge detection method (canny, sobel)
    :param img: image in which to detect cracks
    :return: binary image with cracks detected
    """

    # structure = np.ones((3, 3), dtype=np.int)
    # labeled, ncomponents = label(img, structure)
    # print(f"Scipy components {ncomponents}")
    # print(f"Label: \n{labeled}\n\n")

    height, width = img.shape

    cracks = connected_components(img, method)
    result = np.zeros((height, width))

    if len(cracks) != 0:
        for crack in cracks:
            for i in range(0, len(crack)):
                x, y = crack.pop()
                result[x, y] = 1

        result = result.astype('uint8')
        contours, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            cv.drawContours(result, cnt, -1, (255, 255, 255), 2)

    # np.set_printoptions(threshold=sys.maxsize, formatter={"float": "{: .0f}".format})
    # print(all_component_connected)
    # print(f"CV2 Components {nb_components}")
    # print(f"My components {id_components}")

    return result


def connected_components(img, method):
    r"""
    Detect connected components in an image
    :param method: edge detection method (canny, sobel)
    :param img: image in which to detect connected components
    :return: stack with the coordinates of the detected cracks
    """

    height, width = img.shape

    if method == "Sobel":
        value_found = 0.098
        crack_lenght = 200

    elif method == "Canny":
        value_found = 1
        crack_lenght = 20

    else:
        raise Exception("Specify the edge detection method: Canny or Sobel")

    visited = np.zeros((height, width), dtype=bool)

    tmp_stack = []
    coordinates_result_cracks = []
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

                if lenght_components >= crack_lenght:
                    # Crack cetected
                    coordinates_result_cracks.append(coordinates_current_component.copy())

                coordinates_current_component.clear()

    return coordinates_result_cracks
