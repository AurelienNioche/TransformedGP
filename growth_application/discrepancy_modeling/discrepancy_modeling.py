import torch
import gpytorch
from tqdm.autonotebook import tqdm


class MeanFunc(gpytorch.means.mean.Mean):

    def __init__(self, m, theta):
        super().__init__()
        self.m = m
        self.theta = theta

    def forward(self, x):
        return self.m.forward(x.squeeze(-1), self.theta)


class GP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, m, theta):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MeanFunc(m=m, theta=theta)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DiscrepancyModel:

    def __init__(self, data, m, theta, noise_init_value=50):

        # extract data
        self.train_x = torch.from_numpy(data.age.values).double()
        self.train_y = torch.from_numpy(data.height.values).double()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_init_value

        self.m = m
        self.theta = theta

        self.gp = GP(train_x=self.train_x, train_y=self.train_y,
                     likelihood=self.likelihood, m=self.m, theta=self.theta)

        # for backup
        self.hist_loss = []
        self.hist_noise = []
        self.hist_outputscale = []
        self.hist_lengthscale = []

    def train(
            self,
            learning_rate=0.05,
            epochs=1000,
            seed=123,
            progress_bar=True,
            progress_bar_desc=None):

        # Seed torch
        torch.manual_seed(seed)

        # Find optimal model hyperparameters
        self.gp.train()
        self.likelihood.train()

        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,
                                                       self.gp)

        # for name, p in self.gp.named_parameters():
        #     if p.requires_grad:
        #         print(f"{name}: {p.data.shape}")
        #
        # out:
        # likelihood.noise_covar.raw_noise: torch.Size([1])
        # covar_module.raw_outputscale: torch.Size([])
        # covar_module.base_kernel.raw_lengthscale: torch.Size([1, 1])

        self.hist_loss = []
        self.hist_noise = []
        self.hist_outputscale = []
        self.hist_lengthscale = []

        pbar = tqdm(total=epochs, desc=progress_bar_desc) \
            if progress_bar else None

        for i in range(epochs):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = self.gp(self.train_x)

            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

            # Update progress bar
            if pbar:
                pbar.update()
                pbar.set_postfix(loss=loss.item())

            # For backup
            loss = loss.item()
            noise = self.gp.likelihood.noise_covar.noise.item()
            output_scale = self.gp.covar_module.outputscale.item()
            length_scale = self.gp.covar_module.base_kernel.lengthscale.item()
            self.hist_loss.append(loss)
            self.hist_noise.append(noise)
            self.hist_outputscale.append(output_scale)
            self.hist_lengthscale.append(length_scale)

        if pbar:
            pbar.close()

        return self.hist_loss

    def pred(self, test_x):

        # Get into evaluation (predictive posterior) mode
        self.gp.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions by feeding model through likelihood
            observed_pred = self.likelihood(self.gp(test_x))

            # Get mean
            gp_mean = observed_pred.mean.numpy()

            # Get upper and lower confidence bounds
            gp_std = observed_pred.stddev.numpy()
            gp_lower = gp_mean - 1.96*gp_std
            gp_upper = gp_mean + 1.96*gp_std

        m_pred = self.m.forward(test_x, self.theta).numpy()
        return m_pred, gp_mean, gp_lower, gp_upper
