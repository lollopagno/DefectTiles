import cv2 as cv


def detect(img):
    r"""
    Detects pinhole in the image
    :param img: image in which to detect pinhole
    :return: binary image with pinhole detected
    """
    p_count = 0  # Pinhole count
    c_range = 0  # Range del corner
    e_range = 0  # Range degli edge
    row = 0  # Maximum number of image pixels along any row
    col = 0  # Maximum number of image pixels along any column

    r_left, r_c, r_right = divide(img, 3)


def divide(img, n_portion):
    r"""
    Divide the image into portions
    :param img: Image to split
    :param n_portion: number of portions to create
    :return: portions of the image by number
    """
    height, width = img.shape

    range = round(width / n_portion)
    region_left = img[0:height, 0:range]
    region_central = img[0:height, range:range * 2]
    region_right = img[0:height, range * 2:]

    cv.imshow("L", region_left)
    cv.imshow("C", region_central)
    cv.imshow("R", region_right)

    return region_left, region_central, region_right