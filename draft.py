from sklearn.gaussian_process.kernels import RBF


kernel = RBF()

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel,
                              n_restarts_optimizer=0,
                              alpha=1)

X = np.atleast_2d(x).T

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

x_plot = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x_plot, return_std=True)

# Create "samples of functions"
y_samples = gp.sample_y(x_plot, 10)