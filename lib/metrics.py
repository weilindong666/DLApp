# -*- coding: UTF-8 -*-
'''
@Time    : 2023/1/13 17:52
@Author  : 魏林栋
@Site    : 
@File    : metrics.py
@Software: PyCharm
'''
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.multiclass import type_of_target
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_curve, auc
from sklearn.utils import column_or_1d, check_consistent_length, assert_all_finite


def analysis_pred_binary(y_true, y_score, y_pred=None, alpha=0.95, use_youden=False):
    if isinstance(y_score, (list, tuple)):
        y_score = np.array(y_score)
    y_true = column_or_1d(np.array(y_true))  # 把label作成1维numpy array
    assert sorted(np.unique(y_true)) == [0, 1], f"结果必须是2分类！"
    assert len(y_true) == len(y_score), '样本数必须相等！'
    if len(y_score.shape) == 2:
        y_score = column_or_1d(y_score[:, 1])
    elif len(y_score.shape) > 2:
        raise ValueError(f"y_score不支持>2列的数据！现在是{y_score.shape}")
    else:
        y_score = column_or_1d(y_score)
    tpr, tnr, thres = calc_sens_spec(y_true, y_score)
    if y_pred is None:
        y_pred = np.array(y_score >= (thres if use_youden else 0.5)).astype(int)
    acc = np.sum(y_true == y_pred) / len(y_true)
    tp = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
    tn = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
    fp = np.sum(y_pred[y_true == 0] == 1)
    fn = np.sum(y_pred[y_true == 1] == 0)
    ppv = tp / (tp + fp + 1e-6)
    npv = tn / (tn + fn + 1e-6)
    f1 = 2 * tpr * ppv / (ppv + tpr)
    auc, ci = calc_95_CI(y_true, y_score, alpha=alpha, with_auc=True)
    # print(tp, tn, fp, fn)
    return acc, auc, [float(f"{i_:.6f}") for i_ in ci], tpr, tnr, ppv, npv, ppv, tpr, f1, thres

def calc_95_CI(ground_truth, predictions, alpha=0.95, with_auc: bool = True):
    auc, auc_cov = delong_roc_variance(ground_truth, predictions)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    if with_auc:
        return auc, ci
    else:
        return ci

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count

def calc_sens_spec(y_true, y_score, with_thres=True):
    fpr, tpr, tnr, fnr, thresholds = any_curve(y_true, y_score)
    idx = 0
    maxv = -1e6
    for i, v in enumerate(tpr - fpr):
        if v > maxv:
            maxv = v
            idx = i
    #    idx = np.argmax(tpr - fpr)
    if with_thres:
        return tpr[idx], tnr[idx], thresholds[idx]
    else:
        return tpr[idx], tnr[idx]

def any_curve(y_true, y_score, *, pos_label=None, sample_weight=None):
    fps, tps, tns, fns, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    if fps[-1] <= 0:
        print("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        print("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    if tns[0] <= 0:
        print("No negative samples in y_true, "
                      "true negative value should be meaningless",
                      UndefinedMetricWarning)
        tnr = np.repeat(np.nan, tns.shape)
    else:
        tnr = tns / tns[0]

    if fns[0] <= 0:
        print("No positive samples in y_true, "
                      "false negative value should be meaningless",
                      UndefinedMetricWarning)
        fnr = np.repeat(np.nan, fns.shape)
    else:
        fnr = fns / fns[0]

    return fpr, tpr, tnr, fnr, thresholds


def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.

    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    pos_label : int or str, default=None
        The label of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    tns = fps[-1] - fps
    fns = tps[-1] - tps
    return fps, tps, tns, fns, y_score[threshold_idxs]

def check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.

    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.

    Parameters
    ----------
    pos_label : int, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    pos_label : int
        If `pos_label` can be inferred, it will be returned.

    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
    """
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in 'OUS' or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def draw_roc(y_test, y_score, title='ROC', labels=None):
    """
    绘制ROC曲线
    Args:
        y_test: list或者array，为真实结果。
        y_score: list或者array，为模型预测结果。
        title: 图标题
        labels: 图例名称

    Returns:

    """
    if not isinstance(y_test, (list, tuple)):
        y_test = [y_test]
    if not isinstance(y_score, (list, tuple)):
        y_score = [y_score]
    if labels is None:
        labels = [''] * len(y_score)
    assert len(y_test) == len(y_score) == len(labels)
    colors = ["deeppink", "navy", "aqua", "darkorange", "cornflowerblue"]
    ls = ['-', ':', '--', ':']
    plt.figure()
    for idx, (y_test_, y_score_, label) in enumerate(zip(y_test, y_score, labels)):
        # enc = OneHotEncoder(handle_unknown='ignore')
        # y_test_binary = enc.fit_transform(y_test_.reshape(-1, 1)).toarray()
        y_score_1 = y_score_[:, 1]
        fpr, tpr, _ = roc_curve(y_test_, y_score_1)
        # print(y_test_, y_score_1)
        auc_, ci = calc_95_CI(np.squeeze(y_test_), y_score_1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} AUC: {roc_auc:0.3f} (95%CI {ci[0]:.3f}-{ci[1]:.3f})",
                 color=colors[idx % len(colors)], linestyle=ls[idx % len(ls)], linewidth=4)

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'./result/model_RandomForest_roc.png', bbox_inches='tight')
    return './result/model_RandomForest_roc.png'