import numpy as np
import warnings
import dill
from multiprocessing import Pool, cpu_count
from tqdm.autonotebook import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt

from model.discrepancy import DiscrepancyModel
from model.memory import m, fit_data
from plot.plot import plot_losses, plot_results
from data.get_data import get_data

def run_artificial(
        theta = (0.05, 0.2),
        n_samples = 100,
        learn_inducing_locations = False,
        n_inducing_points = 100,
        epochs = 500,
        learning_rate = 0.001,
        seed=123):

    # Create artificial data
    np.random.seed(seed)
    d = np.random.uniform(0, 2 ** 5, 1000)
    r = np.random.randint(0, 10, size=1000)

    x = np.concatenate((d[None, :], r[None, :]), axis=0).T
    y = (m(x, (0.05, 0.2)) > np.random.random(x.shape[0])).astype(int)

    dm = DiscrepancyModel(
        x=x,
        y=y,
        m=m,
        theta=theta,
        n_samples=n_samples,
        learn_inducing_locations=learn_inducing_locations,
        n_inducing_points=n_inducing_points)

    losses = dm.train(epochs=epochs, learning_rate=learning_rate)

    plot_losses(losses)
    plot_results(dm=dm, theta_truth=theta)

    return losses


def run_apply_async_multiprocessing(
        func, argument_list,
        num_processes=cpu_count()):

    pbar = tqdm(total=len(argument_list))

    pool = Pool(processes=num_processes)

    jobs = [pool.apply_async(func=func,
                             kwds=argument)
            for argument in argument_list]
    pool.close()
    result_list_tqdm = []
    for job in jobs:
        result_list_tqdm.append(job.get())
        pbar.update()
    pbar.close()
    return result_list_tqdm


def run_single_user(
        u, df, epochs, learning_rate,
        ignore_warnings,
        progress_bar=True,
        **other_dm_settings):

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    try:

        x = np.concatenate((df.d.values[None, :], df.r.values[None, :]),
                           axis=0).T
        y = df.y.values

        theta = fit_data(x, y, seed=u)

        dm = DiscrepancyModel(
            x=x,
            y=y,
            m=m,
            theta=theta,
            **other_dm_settings)

        losses = dm.train(epochs=epochs, learning_rate=learning_rate,
                          progress_bar=progress_bar)

        fig, axes = plt.subplots(figsize=(10, 10), nrows=2)
        plot_losses(losses, title=f"User {u}", ax=axes[0])
        plot_results(dm=dm,
                     x_max=1e5 * 4,
                     presentations=[2e5 * i for i in range(1, 3)],
                     title=f"User {u}", ax=axes[1])
        os.makedirs("fig", exist_ok=True)
        plt.savefig(f"fig/user_{u}.pdf")

        return {
            "user": u,
            "dm": dill.dumps(dm),
            "m": m.__name__,
            "theta": theta,
            **other_dm_settings
        }

    except Exception as e:
        print(f"Encountered error with user {u}: {e}")
        return {}


def run_human(ignore_warnings=True):

    data = get_data("data/data_character_meaning.csv")

    settings = dict(
        learning_rate=0.001,
        epochs=int(2e4),
        n_samples=100,
        n_inducing_points=100,
        learn_inducing_locations=False,
        jitter=1e-07)

    users = data.u.unique()

    argument_list = [
        dict(df=data[data.u == u],
             u=u,
             ignore_warnings=ignore_warnings,
             progress_bar=False,
             **settings) for u in users
    ]

    result_list = run_apply_async_multiprocessing(
        func=run_single_user,
        argument_list=argument_list)

    # Saving
    df_dm = pd.DataFrame(result_list)
    path = "bkp/dm_cpc.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


def main():
    # run_artificial()

    run_human()


if __name__ == "__main__":
    main()
