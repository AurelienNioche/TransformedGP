import numpy as np
import pandas as pd
import scipy.special


def generate_data(u, seed=123, max_x=1, n=100, tau=3.333, theta=0.5):
np.random.seed(seed)

data = pd.DataFrame(np.random.uniform(0, 1, size=(n * 10, 4)),
columns=["p0", "x0", "p1", "x1"])
for i in range(2):
data[f"x{i}"] = data[f"x{i}"].values * max_x

data = data[~((data.p0 >= data.p1) & (data.x0 >= data.x1))]
data = data[~((data.p1 >= data.p0) & (data.x1 >= data.x0))]
data = data.sample(n=n, replace=False)

p0 = data.p0.values
p1 = data.p1.values
x0 = data.x0.values
x1 = data.x1.values

seu0 = p0 * u(x0, theta)
seu1 = p1 * u(x1, theta)

diff_eu = seu1 - seu0

p_choice_1 = scipy.special.expit(tau * diff_eu)
choices = np.zeros(n, dtype=int)
choices[:] = p_choice_1 > np.random.random(size=n)
data['choices'] = choices

return data
