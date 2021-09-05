import numpy as np
import scipy.optimize


def fit(model, data, seed):

    def loss(param):
        pred = model.forward(data.age.values, param)
        return np.mean((data.height.values - pred) ** 2)

    np.random.seed(seed)
    res = scipy.optimize.minimize(fun=loss,
                                  x0=model.x0,
                                  method='BFGS')
    return res
