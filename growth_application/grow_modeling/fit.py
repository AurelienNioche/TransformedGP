import numpy as np
from scipy.optimize import minimize


def fit(model, data, seed):

    def loss(param):
        pred = model.forward(data.age.values, param)
        return np.mean((data.height.values - pred) ** 2)

    np.random.seed(seed)
    res = minimize(fun=loss, x0=model.x0, method='BFGS')
    return res
