import numpy as np
import pandas as pd


def get(n_per_group=100, seed=123):

    """
    Create data based on population statistics
    """

    content = ([
        [40, 98, 125, 133, 151, 162],
        [46, 103, 130, 138, 156, 167],
        [48, 107, 135, 145, 164, 173],
        [50, 112, 141, 152, 172, 180],
        [52, 116, 148, 159, 180, 186],
        [54, 120, 153, 167, 188, 192],
        [60, 125, 158, 172, 193, 197]
    ])

    index = [0.00, 0.025, 0.16, 0.50, 0.84, 0.975, 1.00]

    ages = np.array([0, 5, 10, 12, 15, 20])

    pop_param = pd.DataFrame(content, columns=ages, index=index)
    pop_param.index.name = r"P(Y < y)"
    pop_param = pop_param.T

    # y_err = np.abs(
    #     pop_param[[0.160, 0.840]].values.T - pop_param[0.500].values.T)

    # fig, ax = plt.subplots()
    # ax.errorbar(x=ages, y=data[0.500], yerr=y_err, fmt="o")
    # ax.set_xlabel("age")
    # ax.set_ylabel("height")

    np.random.seed(seed)

    obs_age = []
    obs_height = []

    for a in ages:
        obs_height += list(np.random.normal(pop_param.loc[a, 0.500],
                                            pop_param.loc[a, 0.840] -
                                            pop_param.loc[a, 0.500],
                                            size=n_per_group))
        obs_age += [a, ] * n_per_group

    data = pd.DataFrame({"age": obs_age, "height": obs_height})
    return data
