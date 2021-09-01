import dill
import pandas as pd

from accuracy_comparison.accuracy_comparison import accuracy


def main():

    bkp_file = "bkp/dm_artificial_mean_cor=2_seed_data=12345_seed_inference=12345_new.pkl"
    print(f"Loading from {bkp_file}...")
    df_dm = pd.read_pickle(bkp_file)
    df_dm.dm = df_dm.dm.apply(lambda x: dill.loads(x))

    # u_truth = u_pow
    # theta_truth = 0.5
    h_set = "sigmoid", "exp", "identity"
    u_set = "u_pow", "u_lin"

    for u_name in u_set:
        for h in h_set:

            dm = df_dm.loc[(u_name, h)].item()
            print(u_name, h, accuracy(dm))


if __name__ == "__main__":
    main()
