import cv2 as cv
import numpy as np

img = cv.imread("Resources/crackTile.jpg")


def scrool_image(img, name):
    height, width = img.shape
    count_1 = 0
    count_0 = 0
    for i in range(0, height):
        for j in range(0, width):
            if img[i, j] == 1:
                # print(str(img[i, j]) + "\n")
                count_1 += 1
            if img[i, j] == 0:
                # print(str(img[i, j]) + "\n")
                count_0 += 1

    print(f"{name} Num pixel 1:{count_1}")
    print(f"{name} Num pixel 0: {count_0}")


def preprocessing(img):
    # Conversion to grayscale
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Normalization
    img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

    # Median filter
    median = cv.medianBlur(img, 3)  # TODO possibilità di provare anche con 5

    # Edge Detection
    canny = cv.Canny(median, 100, 150)

    sobel_grad_x = cv.Sobel(median, cv.CV_64F, 1, 0, ksize=3)
    sobel_grad_y = cv.Sobel(median, cv.CV_64F, 0, 1, ksize=3)

    sobel_abs_grad_x = cv.convertScaleAbs(sobel_grad_x)
    sobel_abs_grad_y = cv.convertScaleAbs(sobel_grad_y)

    sobel = cv.addWeighted(sobel_abs_grad_x, 0.5, sobel_abs_grad_y, 0.5, 0)

    # print(f"Sobel size: {sobel.shape}")
    # print(f"Canny size: {canny.shape}")
    # scrool_image(sobel, "Sobel")
    # scrool_image(canny, "Canny")
    #
    # print(f"Canny. Min {canny.min()}, max: {canny.max()}")
    # print(f"Sobel. Min {sobel.min()}, max: {sobel.max()}")
    return sobel, canny


cv.imshow("Img original", img)
_, img = preprocessing(img)
cv.imshow("Img result", img)


cv.waitKey(0)
