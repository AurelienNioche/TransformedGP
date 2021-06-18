import torch



def plot_discrepancy(data, model, likelihood, m, theta, title=None):
    # extract data
    train_x = data.age.values
    train_y = data.height.values

    # Pickup testing points
    test_x = torch.linspace(0, train_x.max(), 50).double()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions by feeding model through likelihood
        observed_pred = likelihood(model(test_x))

        # Get upper and lower confidence bounds
        gp_lower, gp_upper = observed_pred.confidence_region()
        gp_lower = gp_lower.numpy()
        gp_upper = gp_upper.numpy()

        # Get mean
        gp_mean = observed_pred.mean.numpy()

        # Convert to numpy for plot
        test_x = test_x.numpy()

        # Get model predictions
        m_pred = m.forward(test_x, theta)

        # Initialize plot
        f, axes = plt.subplots(figsize=(20, 7), ncols=2)

        # On the first axis...
        ax = axes[0]

        # Plot training data as black stars
        h_obs, = ax.plot(train_x, train_y, 'k*', label='Observed data')
        # Plot predictive means as blue line
        h_mean, = ax.plot(test_x, gp_mean, 'C0',
                          label="Model + correction - Mean")
        # Shade between the lower and upper confidence bounds
        h_conf = ax.fill_between(test_x, gp_lower, gp_upper, alpha=0.3,
                                 label="Model + correction - Confidence")

        # Plot model predictions
        h_model = ax.plot(test_x, m_pred, color='C1', ls="--",
                          label="Model alone")

        # Add legend / title / format axis
        ax.legend(handles=[h_obs, h_mean, h_conf], )
        ax.set_xlabel("age")
        ax.set_ylabel("height")
        ax.set_title(title)

        # On the second axis...
        ax = axes[1]

        # Add horizontal line
        ax.hlines(y=0, ls="--", xmin=test_x.min(), xmax=test_x.max(),
                  color="black")

        # Compute correction
        corr = gp_mean - m_pred
        corr_lower = gp_lower - m_pred
        corr_upper = gp_upper - m_pred

        # Plot correction and CI
        ax.plot(test_x, corr, 'C0', label="Correction - Mean")
        ax.fill_between(test_x, corr_lower, corr_upper, alpha=0.3,
                        label="Correction - Confidence")

        # Add legend / title / format axis
        ax.set_xlabel("age")
        ax.set_ylabel("correction (height)")
        ax.legend()
        ax.set_title(title)

    return fig, ax