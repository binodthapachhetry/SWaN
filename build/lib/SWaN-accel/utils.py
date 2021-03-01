import numpy as np
import pandas as pd

def as_float64(X):
    return X.astype(np.float64)

def add_name(X, name):
#     n_dim = X.shape[1]
#     names = [name.upper() + '_' + str(n) for n in range(0, n_dim)]
    if(isinstance(name,list)):
       names = [x.upper() for x in name]
    else:
       names = [name.upper()]
       
    result = pd.DataFrame(data=X, columns=names)
    return result

def vec2colarr(X):
    if len(X.shape) == 1:
        return np.reshape(X, (X.shape[0], 1))
    elif len(X.shape) == 2:
        return X
    elif len(X.shape) > 2:
        raise NotImplementedError('''vec2colarr only accepts 1D or 2D numpy
         array''')

def vec2rowarr(X):
    if len(X.shape) == 1:
        return np.reshape(X, (1, X.shape[0]))
    elif len(X.shape) == 2:
        return X
    elif len(X.shape) > 2:
        raise NotImplementedError('''vec2rowarr only accepts 1D or 2D numpy
         array''')
    else:
        return np.nan

def has_enough_samples(X, threshold=1):
    if len(X.shape) == 1:
        X = vec2colarr(X)
    return X.shape[0] >= threshold

def apply_over_subwins(X, func, subwins, **kwargs):
    win_length = int(np.floor(X.shape[0] / subwins))
    start_index = np.ceil((X.shape[0] % subwins) / 2)
    result = []
    if(win_length <1):
        # print("Data does not have points => number of sub windows")
        result.append(func(X, **kwargs))
    else:
        # print("##########################################################")
        # print(X.shape,win_length,start_index,subwins)
        # print(X[0,:])

        for i in range(0, subwins):
            indices = int(start_index) + np.array(range(
                i * win_length,
                (i + 1) * win_length
            ))
            subwin_X = X[indices, :]
            result.append(func(subwin_X, **kwargs))
    return result
