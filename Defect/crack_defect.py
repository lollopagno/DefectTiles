import numpy as np
import cv2 as cv
import sys
from scipy.ndimage.measurements import label


def detect(img):
    # structure = np.ones((3, 3), dtype=np.int)
    # labeled, ncomponents = label(img, structure)
    # print(f"Scipy components {ncomponents}")
    # print(f"Label: \n{labeled}\n\n")

    height, width = img.shape

    all_component_connected = np.zeros((height, width), dtype=float)  # TODO must be remove at the end the algorithm
    visited = np.zeros((height, width), dtype=bool)

    stack_to_search_component = []
    stack_result_crack = []
    stack_current_component = []

    lenght_components = 0
    max_lenght_components = 0
    id_components = 0  # TODO must be remove at the end the algorithm

    # Implement Depth-first search (DFS)
    for i in range(0, height):
        for j in range(0, width):
            lenght_components = 0

            if visited[i, j]:  # Se l'ho già visitato, continuo
                continue

            elif img[i, j] == 0:
                visited[i, j] = True  # Marco come visitato, non mi interessa è uno 0

            else:
                visited[i, j] = True
                stack_to_search_component.append((i, j))
                id_components += 1

                while len(stack_to_search_component) != 0:

                    x, y = stack_to_search_component.pop()
                    all_component_connected[x, y] = id_components

                    lenght_components += 1
                    stack_current_component.append((x, y))

                    if x - 1 >= 0 and y - 1 >= 0:
                        p1 = img[x - 1, y - 1]
                        if p1 == 1 and not visited[x - 1, y - 1]:
                            stack_to_search_component.append((x - 1, y - 1))
                            visited[x - 1, y - 1] = True

                    if x - 1 >= 0:
                        p2 = img[x - 1, y]
                        if p2 == 1 and not visited[x - 1, y]:
                            stack_to_search_component.append((x - 1, y))
                            visited[x - 1, y] = True

                    if x - 1 >= 0 and y + 1 < width:
                        p3 = img[x - 1, y + 1]
                        if p3 == 1 and not visited[x - 1, y + 1]:
                            stack_to_search_component.append((x - 1, y + 1))
                            visited[x - 1, y + 1] = True

                    if y - 1 >= 0:
                        p4 = img[x, y - 1]
                        if p4 == 1 and not visited[x, y - 1]:
                            stack_to_search_component.append((x, y - 1))
                            visited[x, y - 1] = True

                    if y + 1 < width:
                        p5 = img[x, y + 1]
                        if p5 == 1 and not visited[x, y + 1]:
                            stack_to_search_component.append((x, y + 1))
                            visited[x, y + 1] = True

                    if x + 1 < height and y - 1 >= 0:
                        p6 = img[x + 1, y - 1]
                        if p6 == 1 and not visited[x + 1, y - 1]:
                            stack_to_search_component.append((x + 1, y - 1))
                            visited[x + 1, y - 1] = True

                    if x + 1 < height:
                        p7 = img[x + 1, y]
                        if p7 == 1 and not visited[x + 1, y]:
                            stack_to_search_component.append((x + 1, y))
                            visited[x + 1, y] = True

                    if x + 1 < height and y + 1 < width:
                        p8 = img[x + 1, y + 1]
                        if p8 == 1 and not visited[x + 1, y + 1]:
                            stack_to_search_component.append((x + 1, y + 1))
                            visited[x + 1, y + 1] = True

                if lenght_components > max_lenght_components:
                    #show_image(height, width, stack_current_component, id_components)

                    max_lenght_components = lenght_components
                    stack_result_crack = stack_current_component.copy()
                    stack_current_component.clear()


    result = np.zeros((height, width))

    # TODO impostare la soglia del crack

    if len(stack_result_crack) != 0:
        for i in range(0, len(stack_result_crack)):
            x, y = stack_result_crack.pop()
            result[x, y] = 1

        result = result.astype('uint8')
        contours, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            cv.drawContours(result, cnt, -1, (255, 255, 255), 3)

    # np.set_printoptions(threshold=sys.maxsize, formatter={"float": "{: .0f}".format})
    # print(all_component_connected)
    # img = img.astype('uint8')
    # nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)
    # sizes = stats[:, -1]
    #
    # max_label = 1
    # max_size = sizes[1]
    # for i in range(2, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         max_size = sizes[i]
    #
    # img2 = np.zeros(output.shape)
    # img2[output == max_label] = 255
    # cv.imshow("Biggest component", img2)
    #
    # print(f"CV2 Components {nb_components}")
    # print(f"My components {id_components}")

    return result


def show_image(h, w, point, iteration):
    img = np.zeros((h, w))
    for i in range(0, len(point)):
        x, y = point.pop()
        img[x, y] = 1

    cv.imshow(f"Crack {iteration}", img)
