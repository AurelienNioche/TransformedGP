functions{
  matrix squared_exp(int N, int P, matrix dist_mat, real l, real sigmasq){
    matrix[N,P] ret = sigmasq * exp(-0.5 * dist_mat / l^2);
    return(ret);
  }
  vector M(vector x, real b){
    return 1 - exp(- b * x);
  }
  real M_real(real x, real b){
    return 1 - exp(- b * x);
  }
  real softplus(real x) {
      return log(1 + exp(x));
  }
  vector softplus_vec(vector x) {
      return log(1 + exp(x));
  }
}
data {
  int<lower=0> N;
  int<lower=0> N_eval;
  vector<lower=0, upper=1>[N] p0;
  vector[N] x0;
  vector<lower=0, upper=1>[N] p1;
  vector[N] x1;
  int<lower=0, upper=1> chose1[N];
  real prior_shape;
  real prior_rate;
  // real<lower = 0> l; // possibly parameter
  // real<lower = 0> sigmasq; //possibly parameter
  vector[N_eval] x_eval;
  real jitter;
  //real beta;
}
transformed data {

}
parameters {
  vector[N_eval] z; //inducing points style
  real beta_unc;
  real<lower = 0> l; //hyperparameter
  real<lower = 0> sigmasq; //hyperparameter
}
transformed parameters{
  real beta = softplus(beta_unc);
  vector[N_eval] r;
  matrix[N_eval, N_eval] K;
  cholesky_factor_cov[N_eval] K_chol;
  K = gp_exp_quad_cov(to_array_1d(x_eval), sigmasq, l) + diag_matrix(rep_vector(jitter, N_eval));
  K_chol = cholesky_decompose(K);
  //if not positive, then divergence happens (this restricts that the correction near 0 is really small)
  r = softplus_vec(K_chol * z + rep_vector(0.5, N_eval)); // GP(1, K)
}
model {
  vector[N] seu0;
  vector[N] seu1;
  vector[N] r0_pred;
  vector[N] r1_pred;
  //to the model or params if we wish to infer the hyperparameters
  matrix[N_eval, N_eval] K_inv;
  matrix[N_eval, N_eval] K_chol_inv;
  matrix[N, N_eval] K0;
  matrix[N, N_eval] K1;

  K_inv = inverse_spd(K);
  K0 = gp_exp_quad_cov(to_array_1d(x0), to_array_1d(x_eval), sigmasq, l);
  K1 = gp_exp_quad_cov(to_array_1d(x1), to_array_1d(x_eval), sigmasq, l);

  //could use softplus_vec to map to positive GPs to retain monotonicity (does it go like this?)
  //mean prediction of GP p(f | f_eval) (remember that prior mean is 1.0)
  r0_pred = 1.0 + K0 * K_inv * (r - 1.0);
  r1_pred = 1.0 + K1 * K_inv * (r - 1.0);

  seu0 = p0 .* M(x0, beta) .* r0_pred;
  seu1 = p1 .* M(x1, beta) .* r1_pred;
  //prior
  beta ~ gamma(prior_shape, prior_rate);
  z ~ normal(0,1);//Gp prior
  l ~ inv_gamma(5,6);
  sigmasq ~ normal(0, 2);
  //likelihood
  chose1 ~ bernoulli_logit(seu1 - seu0);
}
