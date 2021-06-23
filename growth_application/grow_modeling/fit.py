import numpy as np
import scipy.optimize


def fit(model, data, verbose=False):

    def loss(param):
        pred = model.forward(data.age, param)
        return np.sum((data.height - pred) ** 2)

    res = scipy.optimize.minimize(fun=loss,
                                  x0=model.x0,
                                  bounds=model.bounds,
                                  method='L-BFGS-')
    if verbose:
        print(res)
    return res.x
