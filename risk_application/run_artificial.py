import os
import pandas as pd
import dill
import torch

from cognitive_modeling.models.utility_models import u_pow, u_lin
from simulation.cpc_like import generate_userdata_cpc_like
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel


def run(seed_data=12345, seed_dm_train=12345):

    mean_correction = 2
    seed_cog_fit = 12345

    u_truth = u_pow
    w_truth = None  # Don't model probability distortion
    tau_truth = 100
    theta_truth = 0.5
    h_set = "sigmoid", "exp", "identity"
    u_set = u_pow, u_lin

    n = 325
    n_samples = 100
    learn_inducing_locations = False
    n_inducing_points = 50
    epochs = 1000
    learning_rate = 0.05
    mean_correction = mean_correction

    # --------------------- #

    path = f"bkp/" \
           f"dm_artificial_" \
           f"mean_cor={mean_correction}_" \
           f"lr={str(learning_rate).split('.')[-1]}_" \
           f"epochs={epochs}_" \
           f"seed_data={seed_data}_" \
           f"seed_cog_fit={seed_cog_fit}_" \
           f"seed_dm_train={seed_dm_train}" \
           ".pkl"

    print(f"Data will be saved as: {path}")

    # -------------------- #

    data = generate_userdata_cpc_like(
        u=u_truth,
        tau=tau_truth,
        theta=theta_truth,
        n=n,
        seed=seed_data)

    opt_param = fit_cpc_like(data, u=u_truth, w=w_truth,
                             seed=seed_cog_fit)
    tau, theta = opt_param

    inducing_points = torch.linspace(0, 1, n_inducing_points)

    n_runs = len(h_set)*len(u_set)

    discrepancy_models = {}

    i = 0
    for h in h_set:
        for u in u_set:

            dm = DiscrepancyModel(
                data=data,
                u=u,
                theta=theta,
                tau=tau,
                h=h,
                n_samples=n_samples,
                learn_inducing_locations=learn_inducing_locations,
                inducing_points=inducing_points,
                mean_correction=mean_correction)

            dm.train(epochs=epochs, learning_rate=learning_rate,
                     seed=seed_dm_train,
                     progress_bar=True,
                     progress_bar_desc=f"Run {i+1}/{n_runs}")

            discrepancy_models[(u.__name__, h)] = dm

            i += 1

    df_dm = pd.DataFrame(discrepancy_models, ["dm"], ).T

    # Saving
    df_dm.dm = df_dm.dm.apply(lambda x: dill.dumps(x))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


def main():
    for seed_dm_train in (1, 12, 123, 12345, 123456):
        for seed_data in (1, 12, 123, 1234, 12345, 123456):
            run(seed_dm_train=seed_dm_train,
                seed_data=seed_data)


if __name__ == "__main__":
    main()
