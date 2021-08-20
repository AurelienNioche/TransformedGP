import numpy as np
import scipy.special
import pandas as pd


def generate_data_cpc_like(u, seed=123, n=100, tau=3.333, theta=0.5):

    np.random.seed(seed)

    data = pd.DataFrame(
        np.zeros((n * 10, 8)),
        columns=[f"p{i}" for i in range(4)] + [f"x{i}" for i in
                                               range(4)])

    j = 0
    for opt in range(2):
        p = np.random.random(size=n * 10)

        data[f'p{j}'] = p
        data[f'p{j + 1}'] = 1 - p

        a = np.random.random(size=(n * 10, 2))
        a = np.sort(-a, -1) * (-1)

        data[f'x{j}'] = a[:, 0]
        data[f'x{j + 1}'] = a[:, 1]

        j += 2

    data = data[~((data.p0 >= data.p2) & (data.x0 >= data.x2))]
    data = data[~((data.p2 >= data.p0) & (data.x2 >= data.x0))]
    data = data.sample(n=n, replace=False)

    pA = data.p0.values
    pB = data.p2.values

    xA0 = data.x0.values
    xA1 = data.x1.values

    xB0 = data.x2.values
    xB1 = data.x3.values

    seuA = pA * u(xA0, theta) + (1 - pA) * u(xA1, theta)
    seuB = pB * u(xB0, theta) + (1 - pB) * u(xB1, theta)

    diff_eu = seuB - seuA

    p_chooseB = scipy.special.expit(tau * diff_eu)
    choices = np.zeros(n, dtype=int)
    choices[:] = p_chooseB > np.random.random(size=n)
    data['choices'] = choices

    return data
