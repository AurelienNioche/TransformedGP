import torch
from torch import nn
from torch import distributions as dist

import warnings


class VariationalStrategy(nn.Module):

    def __init__(self, model, inducing_points, variational_distribution,
                 learn_inducing_locations=True):
        super().__init__()

        # GP model
        self.model = model

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.inducing_points = torch.nn.Parameter(inducing_points)
        else:
            self.inducing_points = inducing_points

        # Variational distribution
        self._variational_distribution = variational_distribution
        self._variational_distribution.initialize_variational_distribution(
            self.prior_distribution)

    def _cholesky_factor(self, induc_induc_covar, jitter=1e-08, max_tries=3):

        A = induc_induc_covar.double()

        try:
            L = torch.cholesky(A, upper=False)

        except RuntimeError as e:
            isnan = torch.isnan(A)
            if isnan.any():
                raise ValueError(
                    f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN.")

            success = False
            Aprime = A.clone()
            jitter_prev = 0
            for i in range(max_tries):
                jitter_new = jitter * (10 ** i)
                Aprime.diagonal(dim1=-2, dim2=-1).add_(
                    jitter_new - jitter_prev)
                jitter_prev = jitter_new
                try:
                    L = torch.cholesky(Aprime, upper=False)
                    warnings.warn(
                        f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal")
                    success = True
                    break
                except RuntimeError:
                    continue
            if not success:
                raise ValueError(
                    f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}. "
                    f"Original error on first attempt: {e}")

        return L

    @property
    def prior_distribution(self):

        shape = self._variational_distribution.shape
        dtype = self._variational_distribution.dtype
        zeros = torch.zeros(shape, dtype=dtype)
        return dist.MultivariateNormal(zeros, torch.eye(shape[0]))

    def kl_divergence(self):
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        """
        return torch.distributions.kl.kl_divergence(self._variational_distribution(),
                                                    self.prior_distribution)

    def forward(self, x, jitter=1e-04):

        # Get p(u)/q(u)
        variational_dist_u = self._variational_distribution()
        inducing_values = variational_dist_u.mean
        variational_inducing_covar = variational_dist_u.covariance_matrix

        # Compute full prior distribution
        full_inputs = torch.cat([self.inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)

        full_covar = full_output.covariance_matrix

        # Covariance terms
        num_induc = self.inducing_points.size(-2)
        num_data = x.shape[0]

        test_mean = full_output.mean[..., num_induc:]

        induc_induc_covar = full_covar[..., :num_induc, :num_induc] \
            + torch.eye(num_induc)*jitter
        induc_data_covar = full_covar[..., :num_induc, num_induc:]

        data_data_covar = full_covar[..., num_induc:, num_induc:] \
            + torch.eye(num_data)*jitter

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        interp_term = torch.triangular_solve(
            input=induc_data_covar.double(), A=L,
            upper=False).solution.to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2)
                           @ inducing_values.unsqueeze(-1)).squeeze(-1) \
            + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.covariance_matrix.mul(-1)

        middle_term = variational_inducing_covar + middle_term

        predictive_covar = (
            data_data_covar
            + interp_term.transpose(-1, -2) @ middle_term @ interp_term
        )

        # Return the distribution
        return dist.MultivariateNormal(predictive_mean, predictive_covar)