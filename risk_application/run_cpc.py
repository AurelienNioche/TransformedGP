import os
import pandas as pd
import dill
from tqdm.autonotebook import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

from cognitive_modeling.models.utility_models import u_pow
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel
from data.data import get

import git


REPO = git.Repo(search_parent_directories=True)


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


def run_single_subject(
        s, d, u, w, h,
        epochs,
        learning_rate,
        seed_cog_fit,
        seed_dm_train,
        silent_error,
        progress_bar=False,
        progress_bar_desc=None,
        **other_dm_settings):

    try:

        opt_param = fit_cpc_like(d, u=u, w=w,
                                 seed=seed_cog_fit)

        tau = opt_param[0]
        theta = opt_param[1]

        dm = DiscrepancyModel(
            data=d,
            u=u,
            theta=theta,
            tau=tau,
            h=h,
            **other_dm_settings)

        dm.train(
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed_dm_train,
            progress_bar=progress_bar,
            progress_bar_desc=progress_bar_desc)

        return {
            "subject": s,
            "dm": dill.dumps(dm),
            "h": h,
            "u": u.__name__,
            "tau": tau,
            "theta": theta,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed_cog_fit": seed_cog_fit,
            "seed_dm_train": seed_dm_train,
            "git_branch": REPO.active_branch.name,
            "git_hash": REPO.head.commit.hexsha,
            **other_dm_settings
        }

    except Exception as e:
        if silent_error:
            print(f"Encountered error with subject {s}: {e}")
            return {}
        else:
            raise e


def run(multiprocess=True,
        seed_dm_train=12345):

    data = get()

    n_inducing_points = 50
    inducing_points = list(np.linspace(0, 1, n_inducing_points+2)[1:-1])

    settings = dict(
        u=u_pow,
        w=None,
        learning_rate=0.05,
        epochs=1000,
        n_samples=100,
        inducing_points=inducing_points,
        learn_inducing_locations=False,
        mean_correction=2,
        jitter=1e-07,
        seed_cog_fit=12345,
        seed_dm_train=seed_dm_train,
        h="sigmoid",
        silent_error=True)

    print("Using settings:")
    print(settings)

    counts = data.subject.value_counts()
    subject_325 = counts[counts == 325].index  # Take subjects with 325 trials

    # ----------------------------- #

    path = f"bkp/dm_cpc_" \
           f"mean_correction={settings['mean_correction']}_" \
           f"lr={str(settings['learning_rate']).split('.')[-1]}_" \
           f"epochs={settings['epochs']}_" \
           f"seed_cog_fit={settings['seed_cog_fit']}_" \
           f"seed_dm_train={settings['seed_dm_train']}" \
           f".pkl"

    print(f"Data will be saved as: {path}")

    # ----------------------------- #

    argument_list = [
        dict(d=data[data.subject == s],
             s=s,
             **settings) for s in subject_325
    ]

    if not multiprocess:
        result_list = []
        for i, arg in enumerate(argument_list):
            r = run_single_subject(progress_bar=True,
                                   progress_bar_desc=f"subject {arg['s']} | global: {i/len(argument_list)*100:.2f}% | user",
                                   **arg)
            result_list.append(r)
    else:
        result_list = run_apply_async_multiprocessing(
            func=run_single_subject,
            argument_list=argument_list)

    # Saving
    df_dm = pd.DataFrame(result_list)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path} (n errors={result_list.count({})})")


def main():

    # for seed_dm_train in (1, 12, 123, 1234, 12345, 123456):
    #     run(seed_dm_train=seed_dm_train)
    run()


if __name__ == "__main__":

    main()
