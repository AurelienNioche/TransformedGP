import os
import pandas as pd
import dill
from tqdm.autonotebook import tqdm
from multiprocessing import Pool, cpu_count

from cognitive_modeling.models.utility_models import u_pow
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling_cpc import DiscrepancyModelCPC
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
        s, d, u, w, h,
        epochs,
        learning_rate,
        seed_cog_fit,
        seed_dm_train,
        silent_error,
        **other_dm_settings):

    try:

        opt_param = fit_cpc_like(d, u=u, w=w,
                                 seed=seed_cog_fit)

        tau = opt_param[0]
        theta = opt_param[1]

        dm = DiscrepancyModelCPC(
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
        if silent_error:
            print(f"Encountered error with subject {s}: {e}")
            return {}
        else:
            raise e


def main():
    multiprocess = True

    mean_correction = 2
    learning_rate = 0.01
    epochs = 2000

    data = get()

    settings = dict(
        u=u_pow,
        w=None,
        learning_rate=learning_rate,
        epochs=epochs,
        n_samples=100,
        n_inducing_points=50,
        learn_inducing_locations=False,
        mean_correction=mean_correction,
        jitter=1e-07,
        seed_cog_fit=12345,
        seed_dm_train=12345,
        h="sigmoid",
        silent_error=False)

    counts = data.subject.value_counts()
    subject_325 = counts[counts == 325].index  # Take subjects with 325 trials

    argument_list = [
        dict(d=data[data.subject == s],
             s=s,
             **settings) for s in subject_325
    ]

    if not multiprocess:
        result_list = []
        for arg in argument_list:
            r = run_single_subject(**arg)
            result_list.append(r)
    else:
        result_list = run_apply_async_multiprocessing(
            func=run_single_subject,
            argument_list=argument_list)

    # Saving
    df_dm = pd.DataFrame(result_list)
    path = f"bkp/dm_cpc_mean_correction={mean_correction}_lr={str(learning_rate).split('.')[-1]}_epochs={epochs}.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


if __name__ == "__main__":
    main()
