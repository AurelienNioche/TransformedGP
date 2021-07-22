import numpy as np

from . utils import safe_exp


class Ssc:

    """
    Shohoji-Sasaki modified by Cole (SSC)

    T. Cole. The use and construction of anthropometric growth reference standards.
    Nutrition research reviews, 6(1):19–50, 1993.
    """

    x0 = 0, 1, 0, 1, 0, 0, 1

    @staticmethod
    def forward(t, param):
        beta0, beta1 = param[:2]
        h1, k, c, r, t_star = np.log(1+safe_exp(param[2:]))
        Wt = safe_exp(-safe_exp(k * (t_star - t)))
        ft = beta0 + beta1 * t - safe_exp(c - r * t)
        pred = 0.1 * (h1 * Wt + ft * (1 - Wt))
        return pred
