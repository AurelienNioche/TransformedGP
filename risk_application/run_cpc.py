import os
import pandas as pd
import dill
from tqdm.autonotebook import tqdm
from multiprocessing import Pool, cpu_count

from cognitive_modeling.models.utility_models import u_pow
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel
from data.data import get


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
        s, d, u, w, h, epochs, learning_rate,
        **other_dm_settings):

    try:

        opt_param = fit_cpc_like(d, u=u, w=w)

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
            progress_bar=False)

        return {
            "subject": s,
            "dm": dill.dumps(dm),
            "h": h,
            "u": u.__name__,
            "tau": tau,
            "theta": theta,
            **other_dm_settings
        }

    except Exception as e:
        print(f"Encountered error with subject {s}: {e}")
        return {}


def main():
    data = get()

    settings = dict(
        u=u_pow,
        w=None,
        learning_rate=0.1,
        epochs=10000,
        n_samples=100,
        n_inducing_points=50,
        learn_inducing_locations=False,
        jitter=1e-07,
        h="sigmoid")

    counts = data.subject.value_counts()
    subject_325 = counts[counts == 325].index  # Take subjects with 325 trials

    argument_list = [
        dict(d=data[data.subject == s],
             s=s,
             **settings) for s in subject_325
    ]

    result_list = run_apply_async_multiprocessing(
        func=run_single_subject,
        argument_list=argument_list)

    # Saving
    df_dm = pd.DataFrame(result_list)
    path = "bkp/dm_cpc.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


if __name__ == "__main__":
    main()
