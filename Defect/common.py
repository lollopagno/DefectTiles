import cv2 as cv
import math

CRACKS = "Cracks"
MIN_DISTANCE_CRACK = 5
RANGE_MIN_CIRCLES_CRACK = 10

MIN_DISTANCE_BLOB = 3
MAX_DISTANCE_BLOB = 20
RANGE_MAX_CIRCLES_BLOB = 6

def calc_distance(contour, defect, min_distance=5, max_distance=7):
    r"""
    Calculate the weighted distance between all points of the contour with its center
    :param contour: current contour
    :param defect: type of defect
    :param min_distance: minimum distance to evaluate the calculation weight to be attributed
    :param max_distance: maximum distance to evaluate the calculation weight to be attributed
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
    range_distance = all_distances_array[-1] - all_distances_array[0]

    if defect == CRACKS:
        return all_distances > MIN_DISTANCE_CRACK and range_distance > RANGE_MIN_CIRCLES_CRACK
    else:
        return MIN_DISTANCE_BLOB <= all_distances <= MAX_DISTANCE_BLOB and 2.5 < range_distance <= RANGE_MAX_CIRCLES_BLOB
