import numpy as np


def numeric_score(prediction, groundtruth):
    r"""
    Calculation prediction.
    """

    prediction[prediction > 0.9] = 1
    prediction[prediction <= 0.9] = 0

    false_positive = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    false_negative = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    true_positive = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    true_negative = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return false_positive, false_negative, true_positive, true_negative


def accuracy(prediction, groundtruth):
    r"""
    Calculation of accuracy.
    """

    false_positive, false_negative, true_positive, true_negative = numeric_score(prediction, groundtruth)
    total_prediction = false_positive + false_negative + true_positive + true_negative
    accuracy = np.divide(true_positive + true_negative, total_prediction)
    return accuracy * 100.0
