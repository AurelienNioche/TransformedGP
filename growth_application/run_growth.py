import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dill

from grow_modeling.models.linf import Linf
from grow_modeling.models.jpps import Jpps
from grow_modeling.models.logf import Logf
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel
from gp_regression.gp_regression import GPRegression

from grow_modeling.fit import fit
from data.data import get_data

sns.set("paper")


def fast_plot(data, m, theta):
    x_plot = np.linspace(data.age.values.min(), data.age.values.max(), 100)
    pred = m.forward(x_plot, theta)

    fig, ax = plt.subplots()
    ax.plot(data.age, data.height, "x", alpha=0.2)
    ax.plot(x_plot, pred, color="C1")

    ax.set_title(m.__name__.upper())
    plt.show()


def main():

    seed_cog_fit = 123
    seed_dm_train = 123

    init_settings = dict(
        noise_init_value=50,
    )
    train_settings = dict(
        epochs=1000,
        learning_rate=0.1,
    )

    result_list = []

    data = get_data()

    model_list = Linf, Logf, Jpps

    i = 0
    for m in model_list:
        print(f"Fitting model {m.__name__}...", end=" ", flush=True)
        res_fit = fit(model=m, data=data, seed=seed_cog_fit)
        print("Done")
        assert res_fit.success
        theta = res_fit.x

        print(f"Computing discrepancy...")
        dm = DiscrepancyModel(
            data=data,
            m=m,
            theta=theta,
            **init_settings)

        dm.train(
            progress_bar=True,
            progress_bar_desc=f"{m.__name__} (model {i+1}/{len(model_list)})",
            seed=seed_dm_train,
            **train_settings)

        result_list.append({
            "dm": dill.dumps(dm),
            "m": m.__name__,
            "theta": theta,
            "seed_cog_fit": seed_cog_fit,
            "seed_dm_train": seed_dm_train,
            **init_settings,
            **train_settings,
        })

        # fast_plot(data, m, theta)
        i += 1

    # Add GP textbook
    print("GP regression...")
    train_settings["epochs"] = 5000
    gpr = GPRegression(
        data=data,
        **init_settings)
    gpr.train(seed=seed_dm_train, **train_settings)
    result_list.append({
        "dm": dill.dumps(gpr),
        "m": "gp",
        "theta": None,
        "seed_cog_fit": None,
        "seed_dm_train": seed_dm_train,
        **init_settings,
        **train_settings,
    })
    print("Done")

    # Saving
    df_dm = pd.DataFrame(result_list)
    path = "bkp/dm_growth.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


if __name__ == "__main__":
    main()
