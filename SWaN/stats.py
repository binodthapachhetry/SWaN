from utils import *
import scipy.stats as scipy_stats

def mean(X):
    X = as_float64(X)
    result = vec2rowarr(np.nanmean(X, axis=0))
    result = add_name(result, mean.__name__)
    return result


def std(X):
    X = as_float64(X)
    result = vec2rowarr(np.nanstd(X, axis=0))
    result = add_name(result, std.__name__)
    return result

def skew(X):
    X = as_float64(X)
    result = vec2rowarr(scipy_stats.skew(X, axis=0))
    result = add_name(result, skew.__name__)
    return result

def kurtosis(X):
    X = as_float64(X)
    result = vec2rowarr(scipy_stats.kurtosis(X, axis=0))
    result = add_name(result, kurtosis.__name__)
    return result

def spearmanr(X):
    corr, _ = scipy_stats.spearmanr(X)
    result = vec2rowarr(corr[np.triu_indices(n=3, k=1)])
    result = add_name(result, spearmanr.__name__)
    return result

def positive_amplitude(X):
    X = as_float64(X)
    result = vec2rowarr(np.nanmax(X, axis=0))
    result = add_name(result, positive_amplitude.__name__)
    return result


def negative_amplitude(X):
    X = as_float64(X)
    result = vec2rowarr(np.nanmin(X, axis=0))
    result = add_name(result, negative_amplitude.__name__)
    return result


def amplitude_range(X):
    X = as_float64(X)
    result = vec2rowarr(positive_amplitude(X).values - negative_amplitude(X).values)
    result = add_name(result, amplitude_range.__name__)
    return result


def amplitude(X):
    X = as_float64(X)
    result = vec2rowarr(np.nanmax(np.abs(X), axis=0))
    result = add_name(result, amplitude.__name__)
    return result


def mean_distance(X):
    '''
    Compute mean distance, the mean of the absolute difference between value
     and mean. Also known as 1st order central moment.

     TODO: Questionable?
    '''

    X = as_float64(X)
    result = mean(np.abs(X - np.repeat(mean(X), X.shape[0], axis=0)), axis=0)
    result = add_name(result, mean_distance.__name__)
    return result

