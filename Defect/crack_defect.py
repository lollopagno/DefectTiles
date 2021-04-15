import numpy as np
import cv2 as cv

# TODO cercare di far filtrare i crack per:
#  1- entropia per individuare i crack neri da quelli bianchi
#  2- componenti molto piccole

RED = np.array([0, 0, 255])
WHITE = np.array([255, 255, 255])


def detect(img_original, img_edge, method):
    r"""
    Detects cracks in the image
    :param img_original: original image in which to draw the defects
    :param img_edge: binary image that contains the edges
    :param method: edge detection method (canny, sobel)
    :return: original image with cracks detected, binary image with cracks detected
    """

    cracks = connected_components(img_edge / 255, method)
    cracks_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)

    if len(cracks) != 0:
        for crack in cracks:
            for i in range(0, len(crack)):
                x, y = crack.pop()
                cracks_detect[x, y] = 1

        contours, hierarchy = cv.findContours(cracks_detect.astype('uint8'), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cracks_detect = np.zeros(img_edge.shape[:2], dtype=np.float64)
        for cnt in contours:

            area = cv.contourArea(cnt)
            if area < 2.0:
                continue

            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.05 * peri, True)
            objCor = len(approx)
            if objCor < 4:
                cv.drawContours(cracks_detect, cnt, -1, (255, 255, 255), 1)
                cv.drawContours(img_original, cnt, -1, (0, 255, 0), 3)

        subtract = cv.subtract(img_edge, cracks_detect, dtype=cv.CV_8U)
    else:
        subtract = np.zeros(img_edge.shape[:2], dtype=np.uint8)

    return img_original, subtract


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
