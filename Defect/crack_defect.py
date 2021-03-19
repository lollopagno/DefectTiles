import numpy as np
import sys
from scipy.ndimage.measurements import label

def detect(img):

    # structure = np.ones((3, 3), dtype=np.int)
    # labeled, ncomponents = label(img, structure)
    # print(f"Scipy components {ncomponents}")
    #print(f"Label: \n{labeled}\n\n")

    height, width = img.shape
    stack = []
    result = np.zeros((height, width), dtype=float)
    visited = np.zeros((height, width), dtype=bool)
    id_counter = 0

    # Implement Depth-first search (DFS)
    for i in range(0, height):
        for j in range(0, width):

            if visited[i, j]:  # Se l'ho già visitato, continuo
                continue

            elif img[i, j] == 0:
                visited[i, j] = True  # Marco come visitato, non mi interessa è uno 0

            else:
                visited[i, j] = True
                stack.append((i, j))
                id_counter += 1

                while len(stack) != 0:

                    x,y = stack.pop()
                    result[x, y] = id_counter

                    if x - 1 >= 0 and y - 1 >= 0:
                        p1 = img[x - 1, y - 1]
                        if p1 == 255 and not visited[x - 1, y - 1]:
                            stack.append((x-1, y-1))
                            visited[x-1, y-1] = True

                    if x - 1 >= 0:
                        p2 = img[x - 1, y]
                        if p2 == 255 and not visited[x - 1, y]:
                            stack.append((x - 1, y))
                            visited[x - 1, y] = True

                    if x - 1 >= 0 and y + 1 < width:
                        p3 = img[x - 1, y + 1]
                        if p3 == 255 and not visited[x - 1, y + 1]:
                            stack.append((x - 1, y + 1))
                            visited[x - 1, y + 1] = True

                    if y - 1 >= 0:
                        p4 = img[x, y - 1]
                        if p4 == 255 and not visited[x, y - 1]:
                            stack.append((x, y - 1))
                            visited[x, y - 1] = True

                    if y + 1 < width:
                        p5 = img[x, y + 1]
                        if p5 == 255 and not visited[x, y + 1]:
                            stack.append((x, y + 1))
                            visited[x, y + 1] = True

                    if x + 1 < height and y - 1 >= 0:
                        p6 = img[x + 1, y - 1]
                        if p6 == 255 and not visited[x + 1, y - 1]:
                            stack.append((x + 1, y - 1))
                            visited[x + 1, y - 1] = True

                    if x + 1 < height:
                        p7 = img[x + 1, y]
                        if p7 == 255 and not visited[x + 1, y]:
                            stack.append((x + 1, y))
                            visited[x + 1, y] = True

                    if x + 1 < height and y + 1 < width:
                        p8 = img[x + 1, y + 1]
                        if p8 == 255 and not visited[x + 1, y + 1]:
                            stack.append((x + 1, y + 1))
                            visited[x + 1, y + 1] = True

    np.set_printoptions(threshold=sys.maxsize, formatter={"float": "{: .0f}".format})
    print(result)

    return result
