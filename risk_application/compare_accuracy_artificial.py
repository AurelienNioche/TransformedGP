from accuracy_comparison.accuracy_comparison import accuracy_comparison_artificial


def main():

    backup_file = \
        "bkp/dm_artificial_mean_cor=2_lr=05_epochs=300_seed_data=12345_seed_cog_fit=12345_seed_dm_train=12345.pkl"

    accuracy_comparison_artificial(
        backup_file=backup_file)


if __name__ == "__main__":
    main()
