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


def accuracy(prediction, target):
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


def IoU(prediction, target):
    r"""
    Calculation of IoU.
    :param target: target input.
    :param prediction: predicted input.
    """

    if target.shape != prediction.shape:
        raise Exception("Input target has dimension ", target.shape, ". Predicted values have shape", prediction.shape)

    if len(target.shape) != 4:
        raise Exception("Input target has dimension ", target.shape, ". Must be 4.")


    prediction[prediction > 0.9] = 1
    prediction[prediction <= 0.9] = 0

    overlap = prediction * target  # Logical AND
    union = prediction + target  # Logical OR

    if union.all() == 0:
        IoU_score = 0
    else:
        IoU_score = np.divide(overlap.sum(), float(union.sum()))

    # intersection = np.logical_and(target, prediction).sum()
    # union = np.logical_or(target, prediction).sum()

    # IoU_sum += IoU_score
    # result_IoU = np.divide(IoU_sum, target.shape[0])

    #print(IOU)
    return IoU_score
