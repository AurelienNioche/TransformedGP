import numpy as np

from . utils import safe_exp


class Pb:

    """
    Preece and Baines (Pb)

    M. Preece and M. Baines. A new family of mathematical models describing
    the human growth curve. Annals of human biology, 5(1):1–24, 1978.

    """

    x0 = np.zeros(5)

    @staticmethod
    def forward(t, param):

        h1, h_star, t_star, s0, s1 = np.log(1+safe_exp(param))

        delta_t = t - t_star
        pred = h1 - 2 * (h1 - h_star) / (
                    safe_exp(s0 * delta_t) + safe_exp(s1 * delta_t))
        return pred
