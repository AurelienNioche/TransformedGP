data {
  int<lower=0> N;
  int<lower=0> N_y;
  vector<lower=0, upper=1>[N] p;
  vector[N] X;
  int<lower=0, upper=1> y[N_y];
  int belong_A0[N_y];
  int belong_A1[N_y];
  int belong_B0[N_y];
  int belong_B1[N_y];
  real tau;
  real jitter;
  vector[N] mu;
  matrix[N, N] kernel_factor;
}
transformed data {
  matrix[N, N] diag_jitter;
  real X_array[N];
  diag_jitter = diag_matrix(rep_vector(jitter, N));
  X_array = to_array_1d(X);
}
parameters {
  real<lower=0> kernel_length;
  real<lower=0> kernel_var; 
  vector[N] eta;
}
transformed parameters{
}
model {
  vector[N] seu;
  vector[N_y] seuA;
  vector[N_y] seuB;
  vector[N_y] diff_eu;
  matrix[N, N] L;
  matrix[N, N] K;
  vector[N] f;
  
  kernel_length ~ normal(1, 1);
  kernel_var ~ normal(1, 1);
  eta ~ std_normal();
  
  K = kernel_factor .* gp_exp_quad_cov(X_array, kernel_var, kernel_length) + diag_jitter;
  L = cholesky_decompose(K);
  
  f = mu + L*eta;
  
  seu = p .* f;
  seuA = seu[belong_A0]+seu[belong_A1];
  seuB = seu[belong_B0]+seu[belong_B1];
  diff_eu = seuB - seuA;
  y ~ bernoulli_logit(tau * diff_eu);
}
generated quantities {
}
