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


def get_accuracy(prediction, target):
    r"""
    Calculation of accuracy.
    :param target: tagret input.
    :param prediction: predicted input.
    :return: system accuracy.
    """

    false_positive, false_negative, true_positive, true_negative = numeric_score(prediction, target)
    total_prediction = false_positive + false_negative + true_positive + true_negative
    acc = np.divide(true_positive + true_negative, total_prediction) * 100

    return acc


def get_IoU(prediction, target, smooth=1e-12):
    r"""
    Calculation of IoU.
    :param target: target input.
    :param prediction: predicted input.
    :param smooth: infinitesimal quantity to avoid division by 0.
    """
    if target.shape != prediction.shape:
        raise Exception("Input target has dimension ", target.shape, ". Predicted values have shape", prediction.shape)

    if len(target.shape) != 4:
        raise Exception("Input target has dimension ", target.shape, ". Must be 4.")

    prediction[prediction > 0.9] = 1
    prediction[prediction <= 0.9] = 0

    intersection = (prediction * target).sum()  # Logical AND
    total = (prediction + target).sum()  # Logical OR
    union = total - intersection

    if union.all() == 0:
        IoU_score = 0.0
    else:
        IoU_score = (intersection + smooth) / (union + smooth)

    return IoU_score * 100
