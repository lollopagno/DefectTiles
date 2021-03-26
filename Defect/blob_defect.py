import numpy as np
import cv2 as cv

def detect(img, size_blob = 20, method="Sobel"):

    # kernel = np.ones((3,3), np.uint8)
    # imgDialation = cv.dilate(img, kernel, iterations=1)
    # cv.imshow("Dialation", imgDialation)

    height, width = img.shape
    start = round(size_blob / 2 + 1)
    result = np.zeros((height, width))

    if method == "Sobel":
        value_found = 0.098

    elif method == "Canny":
        value_found = 1
    else:
        raise Exception("Specify the edge detection method: Canny or Sobel")

    for i in range(start, height - start + 1):
        for j in range(start, width - start + 1):

            if img[i, j] >= value_found:
                detect_blob, blob = calculate_neighbors(pixel=[i, j], img=img, value=value_found, neighbors=start)
                if detect_blob:
                    # Blob detected
                    result = blob.copy()
                    break
    return result


def calculate_neighbors(pixel, img, value, neighbors):
    x, y = pixel
    b_lenght = 0  # Lenght blob

    height, width = img.shape
    result = np.zeros((height, width))

    for n in range(1, neighbors + 1):

        p1 = img[x - n, y - n]
        if p1 >= value:
            result[x - n, y - n] = 1
            b_lenght += 1
        else:
            break

        p2 = img[x - n, y]
        if p2 >= value:
            result[x - n, y] = 1
            b_lenght += 1
        else:
            break

        p3 = img[x - n, y + n]
        if p3 >= value:
            result[x - n, y + n] = 1
            b_lenght += 1
        else:
            break

        p4 = img[x, y - n]
        if p4 >= value:
            result[x, y - n] = 1
            b_lenght += 1
        else:
            break

        p5 = img[x, y + n]
        if p5 >= value:
            result[x, y + n] = 1
            b_lenght += 1
        else:
            break

        p6 = img[x + n, y - n]
        if p6 >= value:
            result[x + n, y - n] = 1
            b_lenght += 1
        else:
            break

        p7 = img[x + n, y]
        if p7 >= value:
            result[x + n, y] = 1
            b_lenght += 1
        else:
            break

        p8 = img[x + n, y + n]
        if p8 >= value:
            result[x + n, y + n] = 1
            b_lenght += 1
        else:
            break

    print(f"Blob lenght: {b_lenght}")
    return b_lenght <= neighbors * neighbors, result
