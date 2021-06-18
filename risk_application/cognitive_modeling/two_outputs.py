import numpy as np
import scipy.special
import scipy.optimize

from . models.utility_models import u_pow


def softplus(x): return np.log(1 + np.exp(x))


def objective(param, data, u_m):
    param = softplus(param)  # All parameters all supposed to be R+

    tau = param[0]
    theta = param[1]

    p0 = data.p0.values
    p1 = data.p1.values
    x0 = data.x0.values
    x1 = data.x1.values
    y = data.choices.values

    seu0 = p0 * u_m(x0, theta)
    seu1 = p1 * u_m(x1, theta)

    diff_eu = seu1 - seu0

    p_choice_1 = scipy.special.expit(tau * diff_eu)  # p choose 1
    p_choice_y = p_choice_1 ** y * (1 - p_choice_1) ** (1 - y)
    return - np.log(p_choice_y).sum()


def optimize(data, u_m=u_pow, x0=None):
    if x0 is None:
        x0 = (0.0, 0.0)  # Assume two parameters
    opt = scipy.optimize.minimize(objective, x0=x0, args=(data, u_m))
    return softplus(opt.x)
