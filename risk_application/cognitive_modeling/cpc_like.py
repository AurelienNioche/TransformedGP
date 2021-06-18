import numpy as np
import scipy.special
import scipy.optimize


from . models.utility_models import u_pow
from . utils import softplus


def fit_cpc_like(data, u=u_pow, w=None, seed=12345):

    def objective(data, u, w):
        pA = data.p0.values
        pB = data.p2.values

        xA0 = data.x0.values
        xA1 = data.x1.values

        xB0 = data.x2.values
        xB1 = data.x3.values

        y = data.choices.values

        def run(param):

            tau = softplus(param[0])
            theta_u = softplus(param[1])

            if w is not None:
                theta_w = scipy.special.expit(param[2])
                wpA = w(pA, theta_w)
                wpB = w(pB, theta_w)

            else:
                wpA = pA
                wpB = pB

            uxA0 = u(xA0, theta_u)
            uxA1 = u(xA1, theta_u)
            uxB0 = u(xB0, theta_u)
            uxB1 = u(xB1, theta_u)

            seuA = wpA * uxA0 + (1 - wpA) * uxA1
            seuB = wpB * uxB0 + (1 - wpB) * uxB1

            diff_seu = seuB - seuA

            p_choice_B = scipy.special.expit(tau * diff_seu)
            p_choice_y = p_choice_B ** y * (1 - p_choice_B) ** (1 - y)

            lls = np.log(p_choice_y + np.finfo(float).eps).sum()
            return - lls

        return run

    np.random.seed(seed)
    if w is None:
        opt = scipy.optimize.minimize(objective(data, u, w), x0=np.ones(2))
        theta_w = None
    else:
        opt = scipy.optimize.minimize(objective(data, u, w), x0=np.ones(3))
        theta_w = scipy.special.expit(opt.x[2])

    tau = softplus(opt.x[0])
    theta_u = softplus(opt.x[1])
    return tau, theta_u, theta_w
