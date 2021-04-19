import cv2 as cv
import math

CRACKS = "Cracks"


def calc_distance(contour, defect, min_distance=7):
    r"""
    Calculate the weighted distance between all points of the contour with its center
    :param contour: current contour
    :param defect: type of defect
    :param min_distance: minimum distance to evaluate the calculation weight to be attributed
    :return: true if the all distances is greater or less than the minimum distance
    """

    try:
        all_distances = 0  # Total distance of all points of the polygon from the center
        all_weights = 0  # Total weight of all points of the polygon from the center

        moment = cv.moments(contour)
        center = (int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00']))

        for item in contour:
            # Calculate the weighted distance
            distance = math.sqrt((math.pow(center[0] - item[0][0], 2) + math.pow(center[1] - item[0][1], 2)))

            if defect == CRACKS:
                weight = 1 if distance > min_distance else 2
            else:
                weight = 1 if distance < min_distance else 2

            all_weights += weight
            all_distances += distance * weight

        all_distances = round(all_distances / all_weights)  # Weighted distance of the polygon
    except:
        return False

    if defect == CRACKS:
        return all_distances >= min_distance
    else:
        return all_distances < min_distance
