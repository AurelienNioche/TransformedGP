import os
import pandas as pd
import dill
import torch

from cognitive_modeling.models.utility_models import u_pow, u_lin
from simulation.cpc_like import generate_data_cpc_like
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel


def run(mean_correction=2,
        seed_data=12345,
        seed_inference=12345):

    seed_data = seed_data
    seed_cog_fit = seed_inference
    seed_dm_train = seed_inference

    u_truth = u_pow
    w_truth = None  # Don't model probability distortion
    tau_truth = 100
    theta_truth = 0.5
    h_set = "sigmoid", "exp", "identity"
    u_set = u_pow, u_lin

    n = 350
    n_samples = 100
    learn_inducing_locations = False
    n_inducing_points = 50
    epochs = 300
    learning_rate = 0.05
    mean_correction = mean_correction

    data = generate_data_cpc_like(
        u=u_truth,
        tau=tau_truth,
        theta=theta_truth,
        n=n,
        seed=seed_data)

    opt_param = fit_cpc_like(data, u=u_truth, w=w_truth,
                             seed=seed_cog_fit)
    tau, theta = opt_param

    inducing_points = torch.linspace(0, 1, n_inducing_points)

    discrepancy_models = {}

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
                     seed=seed_dm_train)

            discrepancy_models[(u.__name__, h)] = dm

    df_dm = pd.DataFrame(discrepancy_models, ["dm"], ).T

    # Saving
    df_dm.dm = df_dm.dm.apply(lambda x: dill.dumps(x))
    path = f"bkp/" \
           f"dm_artificial_" \
           f"mean_cor={mean_correction}"\
           f"_seed_data={seed_data}" \
           f"_seed_inference={seed_inference}" \
           "_new.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


def main():
    # for mean_correction in (0, 1, 2,):
    #     for seed_data in (1, 12, 123, 12345, 123456, 1234567):
    #         for seed_inference in (1, 12, 123, 12345, 123456, 1234567):
    #             run(
    #                 seed_data=seed_data,
    #                 seed_inference=seed_inference,
    #                 mean_correction=mean_correction)
    run()


if __name__ == "__main__":
    main()
