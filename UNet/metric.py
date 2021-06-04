import numpy as np


def numeric_score(prediction, ground_truth):
    r"""
    Calculation prediction.
    """

    prediction[prediction > 0.9] = 1
    prediction[prediction <= 0.9] = 0

    false_positive = np.float(np.sum((prediction == 1) & (ground_truth == 0)))
    false_negative = np.float(np.sum((prediction == 0) & (ground_truth == 1)))
    true_positive = np.float(np.sum((prediction == 1) & (ground_truth == 1)))
    true_negative = np.float(np.sum((prediction == 0) & (ground_truth == 0)))

    return false_positive, false_negative, true_positive, true_negative


def accuracy(prediction, ground_truth):
    r"""
    Calculation of accuracy.
    :return: system accuracy.
    """

    false_positive, false_negative, true_positive, true_negative = numeric_score(prediction, ground_truth)
    total_prediction = false_positive + false_negative + true_positive + true_negative
    acc = np.divide(true_positive + true_negative, total_prediction) * 100

    return acc

