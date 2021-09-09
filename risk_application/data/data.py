import pandas as pd
import numpy as np


def get(file="data/cpc2018.csv"):

    df = pd.read_csv(file)
    data = df[(df.LotNumB == 1) & (df.LotNumA == 1) & (df.Amb == 0)
              & (df.Ha >= 0) & (df.Hb >= 0) & (df.La >= 0) & (df.Lb >= 0)]

    data = pd.DataFrame({
        "subject": data.SubjID,
        "p0": data.pHa.values,
        "x0": data.Ha.values,
        "p1": 1 - data.pHa.values,
        "x1": data.La.values,
        "p2": data.pHb.values,
        "x2": data.Hb.values,
        "p3": 1 - data.pHb.values,
        "x3": data.Lb.values,
        "choices": data.B.values
    })

    # normalize reward
    max_x = np.max(np.concatenate([data[f'x{i}'] for i in range(4)]))
    for i in range(4):
        data[f'x{i}'] = data[f'x{i}'] / max_x

    # n_output_total = len(
    #     [c for c in data.columns if c.startswith("x")])
    #
    # x = np.hstack(
    #     [data[f"x{i}"].values for i in range(n_output_total)])
    # print(np.unique(x, return_inverse=))

    return data
