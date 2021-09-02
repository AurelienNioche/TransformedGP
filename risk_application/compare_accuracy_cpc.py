from accuracy_comparison.accuracy_comparison import accuracy_comparison_cpc


def main():
    bf = "bkp/dm_cpc_new_mean_correction=2_lr=05_epochs=300.pkl"
    accuracy_comparison_cpc(backup_file=bf)


if __name__ == "__main__":
    main()
