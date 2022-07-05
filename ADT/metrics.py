import numpy as np

from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def sensivity_specifity_cutoff(y_true, y_score):
    """Find data-driven cut-off for classification. Cut-off is determined using Youden's index defined as
    sensitivity + specificity - 1.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
    """
    if y_true.shape != y_score.shape:
        raise AssertionError('y_true and y_score do not have the same shape.')

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Remove the first elements in the arrays since `thresholds[0]` represents no instances being predicted
    # and is arbitrarily set to `max(y_score) + 1`.
    fpr = fpr[1:]
    tpr = tpr[1:]
    thresholds = thresholds[1:]

    youdens_indices = tpr - fpr
    threshold_with_max_youdens_idx = thresholds[np.argmax(youdens_indices)]

    return threshold_with_max_youdens_idx


def sensivity_calibrated_cutoff(y_true, y_score, desired_sensitivity):
    """Find cut-off for classification calibrated to desired sensitivity.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
    desired_sensitivity : float between 0 and 1
    """
    if y_true.shape != y_score.shape:
        raise AssertionError('y_true and y_score do not have the same shape.')
    if desired_sensitivity <= 0 or desired_sensitivity >= 1:
        raise ValueError('desired_sensitivity must be between 0 an 1.')

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Remove the first elements in the arrays since `thresholds[0]` represents no instances being predicted
    # and is arbitrarily set to `max(y_score) + 1`.
    fpr = fpr[1:]
    tpr = tpr[1:]
    thresholds = thresholds[1:]

    differences_from_desired_sensitivity = np.abs(desired_sensitivity - tpr)
    print('diff from sens:', differences_from_desired_sensitivity)
    threshold_with_desired_sensitivity = thresholds[np.argmin(differences_from_desired_sensitivity)]

    return threshold_with_desired_sensitivity


def specifity_calibrated_cutoff(y_true, y_score, desired_specificity):
    """Find cut-off for classification calibrated to desired specificity.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
    desired_specificity : float between 0 and 1
    """
    if y_true.shape != y_score.shape:
        raise AssertionError('y_true and y_score do not have the same shape.')
    if desired_specificity <= 0 or desired_specificity >= 1:
        raise ValueError('desired_specificity must be between 0 an 1.')

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Remove the first elements in the arrays since `thresholds[0]` represents no instances being predicted
    # and is arbitrarily set to `max(y_score) + 1`.
    fpr = fpr[1:]
    tpr = tpr[1:]
    thresholds = thresholds[1:]

    differences_from_desired_specificity = np.abs(desired_specificity - (1 - fpr))
    print('spec:', 1 - fpr)
    print('diff from spec:', differences_from_desired_specificity)
    threshold_with_desired_specificity = thresholds[np.argmin(differences_from_desired_specificity)]

    return threshold_with_desired_specificity


if __name__ == '__main__':
    y = np.array([0, 0, 1, 1, 1, 0, 0])
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.5, 0.7])
    fpr, tpr, thres = roc_curve(y, scores)
    print('fpr:', fpr)
    print('tpr:', tpr)
    print('thres:', thres)

    # Youdens's index calibration
    opt_threshold = sensivity_specifity_cutoff(y, scores)
    print('Youden\'s index calibrated threshold:', opt_threshold)

    # sensitivity calibration
    desired_sens = 0.43
    opt_threshold = sensivity_calibrated_cutoff(y, scores, desired_sens)
    print('Sensitivity calibrated threshold:', opt_threshold)

    # specificity calibration
    desired_spec = 0.86
    opt_threshold = specifity_calibrated_cutoff(y, scores, desired_spec)
    print('Specificity calibrated threshold:', opt_threshold)
