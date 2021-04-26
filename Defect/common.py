import cv2 as cv
import math

CRACKS = "Cracks"
MIN_DISTANCE_CRACK = 5
RANGE_MIN_CIRCLES_CRACK = 10

MIN_DISTANCE_BLOB = 0  # 3
MAX_DISTANCE_BLOB = 20  # 10  # 20
RANGE_MAX_CIRCLES_BLOB = 6


def calc_distance(contour, defect):
    r"""
    Calculate the weighted distance between all points of the contour with its center
    :param contour: current contour
    :param defect: type of defect
    :return: true if the all distances is greater or less than the minimum distance
    """

    try:
        all_distances_array = []
        all_distances = 0  # Total distance of all points of the polygon from the center
        all_weights = 0  # Total weight of all points of the polygon from the center

        moment = cv.moments(contour)
        center = (int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00']))

        for item in contour:
            # Calculate the weighted distance
            distance = math.sqrt((math.pow(center[0] - item[0][0], 2) + math.pow(center[1] - item[0][1], 2)))

            if defect == CRACKS:
                weight = 1 if distance > MIN_DISTANCE_CRACK else 2
            else:
                weight = 1 if MIN_DISTANCE_BLOB <= distance <= MAX_DISTANCE_BLOB else 2

            all_weights += weight
            all_distances += distance * weight
            all_distances_array.append(distance * weight)

        all_distances = round(all_distances / all_weights)  # Weighted distance of the polygon
    except:
        return False

    all_distances_array.sort()
    range_distance = round(all_distances_array[-1] - all_distances_array[0])

    if defect == CRACKS:
        return all_distances > MIN_DISTANCE_CRACK and range_distance > RANGE_MIN_CIRCLES_CRACK

    else:
        first_item = all_distances_array[0]
        if 0 <= range_distance <= 3 and 1 <= all_distances <= 9:
            # Potential circle
            isCircle = all_distances - 2 <= first_item
            return isCircle
        else:
            # Potential ellipse
            isEllipse = all_distances - 2 >= first_item
            return isEllipse
