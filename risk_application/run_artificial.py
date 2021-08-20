import os
import pandas as pd
import dill

from cognitive_modeling.models.utility_models import u_pow, u_lin
from simulation.cpc_like import generate_data_cpc_like
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel


def main():

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
    epochs = 1000  # 300
    learning_rate = 0.01  # 0.05
    use_mean_correction = True

    data = generate_data_cpc_like(
        u=u_truth,
        tau=tau_truth,
        theta=theta_truth,
        n=n)

    opt_param = fit_cpc_like(data, u=u_truth, w=w_truth)
    tau, theta = opt_param

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
                n_inducing_points=n_inducing_points,
                use_mean_correction=use_mean_correction)

            dm.train(epochs=epochs, learning_rate=learning_rate)

            discrepancy_models[(u.__name__, h)] = dm

    df_dm = pd.DataFrame(discrepancy_models, ["dm"], ).T

    # Saving
    df_dm.dm = df_dm.dm.apply(lambda x: dill.dumps(x))
    path = "bkp/dm_artificial.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


if __name__ == "__main__":
    main()

