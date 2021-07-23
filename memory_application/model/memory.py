import numpy as np
import scipy.optimize


def m(x, theta):
    delta, rep = x.T
    alpha, beta = theta
    return np.exp(-alpha*(1-beta)**rep*delta)


def fit_data(x, y, seed=123):
    def obj(theta):

        p_success = m(x, theta)
        p_y = p_success**y*(1-p_success)**(1-y)
        logp = np.log(p_y + 1e-07)
        return - logp.sum()

    np.random.seed(seed)
    res = scipy.optimize.minimize(
        obj,
        x0=np.array([0, 0.5]),
        bounds=((0, np.inf), (0, 1)))

    return res.x
