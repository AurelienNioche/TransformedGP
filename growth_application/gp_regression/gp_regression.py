import torch
import gpytorch
from tqdm.autonotebook import tqdm
import numpy as np


class GP(gpytorch.models.ExactGP):
    """
    straight from the doc
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPRegression:

    def __init__(self, data, noise_init_value):

        # extract data
        self.train_x = torch.from_numpy(data.age.values).double()
        self.train_y = torch.from_numpy(data.height.values).double()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_init_value

        self.gp = GP(train_x=self.train_x, train_y=self.train_y,
                     likelihood=self.likelihood)

        self.hist_loss = []

    def train(self, seed=123, epochs=100, learning_rate=0.1):

        # Seed torch
        torch.random.manual_seed(seed)

        # Find optimal model hyperparameters
        self.gp.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.gp.parameters(),
                                     lr=learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.gp)

        self.hist_loss = []

        with tqdm(total=epochs) as pbar:
            for i in range(epochs):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.gp(self.train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, self.train_y)
                loss.backward()

                optimizer.step()

                pbar.update()
                pbar.set_postfix(loss=loss.item())

                self.hist_loss.append(loss.item())

        return self.hist_loss

    def predict(self, test_x):

        # Get into evaluation (predictive posterior) mode
        self.gp.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions by feeding model through likelihood
            observed_pred = self.likelihood(self.gp(test_x))

            # Get upper and lower confidence bounds
            gp_lower, gp_upper = observed_pred.confidence_region()
            gp_lower = gp_lower.numpy()
            gp_upper = gp_upper.numpy()

            # Get mean
            gp_mean = observed_pred.mean.numpy()

        return np.zeros(len(test_x)), gp_mean, gp_lower, gp_upper
