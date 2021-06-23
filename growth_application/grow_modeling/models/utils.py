import numpy as np

MAX = np.log(np.finfo(float).max)


def safe_exp(x):
    return np.exp(np.clip(x, -np.inf, MAX))
