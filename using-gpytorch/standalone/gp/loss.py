class VariationalELBO:

    def __init__(self, likelihood, model, num_data, beta=1.0):
        self.likelihood = likelihood
        self.model = model

        self.num_data = num_data
        self.beta = beta

    def _log_likelihood_term(self, variational_dist_f, target):
        return self.likelihood.expected_log_prob(target, variational_dist_f).sum(-1)

    def __call__(self, variational_dist_f, target):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`approximate_dist_f` (:obj: torch.distribution.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
        """
        # Get likelihood term and KL term
        num_batch = variational_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(variational_dist_f, target).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)
        
        # We assume a uniform prior here
        return log_likelihood - kl_divergence   # + log_prior