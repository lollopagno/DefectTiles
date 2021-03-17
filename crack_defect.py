import numpy as np

def detect(img):
    height, width = img.shape
    stack = []
    result = np.zeros((height, width), dtype=float)
    visited = np.zeros((height, width), dtype=bool)
    id_counter = 1

    # Implement Depth-first search (DFS)
    for x in range(0, height):
        for y in range(0, width):
            if img[x, y] == 0:
                visited[x, y] = True  # Marco come visitato, non mi interessa è uno 0

            elif visited[x, y]:  # Se l'ho già visitato, continuo
                continue

            else:
                stack.append((x, y))

                while len(stack) != 0:

                    x,y = stack.pop()
                    visited[x, y] = True
                    result[x, y] = id_counter

                    if x - 1 >= 0 and y - 1 >= 0:
                        p1 = img[x - 1, y - 1]
                        if p1 == 1 and not visited[x - 1, y - 1]:
                            stack.append(p1)

                    if x - 1 >= 0:
                        p2 = img[x - 1, y]
                        if p2 == 1 and not visited[x - 1, y]:
                            stack.append(p2)

                    if x - 1 >= 0 and y + 1 < width:
                        p3 = img[x - 1, y + 1]
                        if p3 == 1 and not visited[x - 1, y + 1]:
                            stack.append(p3)

                    if y - 1 >= 0:
                        p4 = img[x, y - 1]
                        if p4 == 1 and not visited[x, y - 1]:
                            stack.append(p4)

                    if y + 1 < width:
                        p5 = img[x, y + 1]
                        if p5 == 1 and not visited[x, y + 1]:
                            stack.append(p5)

                    if x + 1 < height and y - 1 >= 0:
                        p6 = img[x + 1, y - 1]
                        if p6 == 1 and not visited[x + 1, y - 1]:
                            stack.append(p6)

                    if x + 1 < height:
                        p7 = img[x + 1, y]
                        if p7 == 1 and not visited[x + 1, y]:
                            stack.append(p7)

                    if x + 1 < height and y + 1 < width:
                        p8 = img[x + 1, y + 1]
                        if p8 == 1 and not visited[x + 1, y + 1]:
                            stack.append(p8)

                if len(stack) == 0:
                    id_counter += 1

    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in result]))

    return result
