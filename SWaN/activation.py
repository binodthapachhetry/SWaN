import numpy as np
from utils import *


def active_perc(X, threshold=0.2):
    """
    The percentage of active samples, active samples are samples whose value is
     beyond certain threshold. Default is 0.2g.
    """
    X = as_float64(X)
    thres_X = X >= threshold
    active_samples = np.sum(thres_X, axis=0)
    result = vec2rowarr(
        active_samples / np.float(thres_X.shape[0]))
    result = add_name(result, active_perc.__name__)
    return result


def activation_count(X, threshold=0.2):
    """
    The number of times signal go across up the active threshold
    """
    X = as_float64(X)
    thres_X = X >= threshold
    active_samples = np.sum(thres_X, axis=0)
    active_crossings_X = np.diff(
        np.insert(thres_X, 0, np.zeros([1, X.shape[1]]), axis=0),
        axis=0) > 0
    active_crossings = np.sum(active_crossings_X, axis=0)
    result = vec2rowarr(np.divide(active_crossings, active_samples))
    result = add_name(result, activation_count.__name__)
    return result


def activation_std(X, threshold=0.2):
    """
    The standard deviation of the durations of actived durations
    """
    X = as_float64(X)
    thres_X = X >= threshold
    cumsum_X = np.cumsum(thres_X, axis=0)
    rise_marker_X = np.diff(
        np.insert(thres_X, 0, np.zeros([1, X.shape[1]]), axis=0),
        axis=0) > 0
    active_crossings = np.sum(rise_marker_X, axis=0)
    zero_marker = active_crossings <= 2
    fall_marker_X = np.diff(
        np.append(thres_X, np.zeros([1, X.shape[1]]), axis=0), axis=0) < 0
    rise_X = np.sort(np.multiply(
        cumsum_X, rise_marker_X, dtype=np.float), axis=0)
    fall_X = np.sort(np.multiply(
        cumsum_X, fall_marker_X, dtype=np.float), axis=0)
    activation_dur_X = fall_X - rise_X + 1
    activation_dur_X[activation_dur_X == 1.] = np.nan
    result = np.nanstd(activation_dur_X, axis=0)
    result[zero_marker] = 0
    result = vec2rowarr(result / X.shape[0])
    result = add_name(result, activation_std.__name__)
    return(result)
