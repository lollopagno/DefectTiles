import cv2 as cv
import numpy as np
import math

CRACKS = "Cracks"
MIN_DISTANCE_CRACK = 5
RANGE_MIN_RADIUS_CRACK = 10


def calc_distance(contour, defect):
    r"""
    Calculate the weighted distance between all points of the contour with its center
    :param contour: current contour
    :param defect: type of defect
    :return: true if the all distances is greater or less than the minimum distance
    """

    try:
        all_rays_array = []
        all_rays = 0  # Total distance of all points of the polygon from the center

        moment = cv.moments(contour)
        center = (int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00']))

        for item in contour:
            # Calculate the weighted distance
            radius = np.sqrt((math.pow(center[0] - item[0][0], 2) + math.pow(center[1] - item[0][1], 2)))

            all_rays += radius
            all_rays_array.append(radius)

        average_radius_contour = all_rays / len(all_rays_array)  # Weighted distance of the polygon
    except Exception:
        return False

    all_rays_array.sort()
    range_radius = round(all_rays_array[-1] - all_rays_array[0])

    if defect == CRACKS:
        return average_radius_contour > MIN_DISTANCE_CRACK and range_radius > RANGE_MIN_RADIUS_CRACK

    else:
        _, radius = cv.minEnclosingCircle(contour)
        maxRadius = radius + 1
        minRadius = radius - 1

        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, -1)
        circularity = (4 * math.pi * area) / math.pow(perimeter, 2)

        print(circularity)
        if minRadius <= average_radius_contour <= maxRadius or circularity >= 0.8:
            # Circle detected
            return True

        else:
            # Potential ellipse
            ellipse = cv.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = round(np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2), 2)
            print(eccentricity)
            # if radius - average_radius_contour <= 2:
            #     return eccentricity >= 0.8
            if radius - average_radius_contour <= 3.5:
                return eccentricity >= 0.7  # TODO per ricerca più ristretta mettere 0.8
