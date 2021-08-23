import os
import pandas as pd
import dill

from cognitive_modeling.models.utility_models import u_pow, u_lin
from simulation.cpc_like import generate_data_cpc_like
from cognitive_modeling.cpc_like import fit_cpc_like
from discrepancy_modeling.discrepancy_modeling import DiscrepancyModel


def main():

    seed_generate = 12345
    seed_cog_fit = 12345
    seed_dm_train = 12345

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
    use_mean_correction = True

    data = generate_data_cpc_like(
        u=u_truth,
        tau=tau_truth,
        theta=theta_truth,
        n=n,
        seed=seed_generate)

    opt_param = fit_cpc_like(data, u=u_truth, w=w_truth,
                             seed=seed_cog_fit)
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

            dm.train(epochs=epochs, learning_rate=learning_rate,
                     seed=seed_dm_train)

            discrepancy_models[(u.__name__, h)] = dm

    df_dm = pd.DataFrame(discrepancy_models, ["dm"], ).T

    # Saving
    df_dm.dm = df_dm.dm.apply(lambda x: dill.dumps(x))
    path = f"bkp/" \
           f"dm_artificial" \
           f"{'_mean_corrected' if use_mean_correction else ''}" \
           f".pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_dm.to_pickle(path)

    print(f"Results saved as: {path}")


if __name__ == "__main__":
    main()

