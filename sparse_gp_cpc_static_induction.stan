functions {
  vector u_pow(vector x, real theta) {
    return pow(x, theta); 
  }
} 

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
  real theta;
  real tau;
  real jitter;
  real kernel_var_prior_mu;
  real kernel_var_prior_std;
  real kernel_length_prior_mu;
  real kernel_length_prior_std;
  int M;
  vector<lower=0, upper=1>[M] Xm;
  int<lower=0, upper=1> u_model_index;
}
transformed data {
  matrix[M, M] diag_jitter;
  diag_jitter = diag_matrix(rep_vector(jitter, M));
  real X_array[N];
  X_array = to_array_1d(X);
}
parameters {
  real<lower=0> kernel_length;
  real<lower=0> kernel_var; 
  vector[M] eta;
}
transformed parameters{
}
model {
  vector[N] seu;
  vector[N_y] seuA;
  vector[N_y] seuB;
  vector[N] Mp;
  vector[N] f_predict;
  vector[N_y] diff_eu;
  matrix[M, M] L;
  matrix[M, M] K;
  vector[M] f;
  matrix[M, N] K_mp;
  vector[M] Mm;
  vector[M] L_dot_eta;
  matrix[M, N] A;
  vector[M] v;
  
  matrix[M, M] K_factor;
  matrix[M, N] K_mp_factor;
  
  kernel_length ~ normal(kernel_length_prior_mu, kernel_length_prior_std);
  kernel_var ~ normal(kernel_var_prior_mu, kernel_var_prior_std);
  eta ~ std_normal();
  
  if (u_model_index == 0) {
      Mm = Xm;
      Mp = X;
  }
  else if (u_model_index == 1) {
      Mm = u_pow(Xm, theta);
      Mp = u_pow(X, theta);
  }
  else {
      reject("u_model_index incorrect", u_model_index);
  }
  Mm = u_pow(Xm, theta);
  Mp = u_pow(X, theta);
  
  K_factor = Mm*Mm';
  K_mp_factor = Mm*Mp';

  K = K_factor .* gp_exp_quad_cov(to_array_1d(Xm), kernel_var, kernel_length) + diag_jitter;
  L = cholesky_decompose(K);

  L_dot_eta = L*eta;
  f = Mm + L_dot_eta;

  K_mp = K_mp_factor .* gp_exp_quad_cov(to_array_1d(Xm), X_array, kernel_var, kernel_length);
  A = mdivide_left_tri_low(L, K_mp);
  v = mdivide_left_tri_low(L, L_dot_eta);

  f_predict = Mp + A' * v;

  seu = p .* f_predict;
  seuA = seu[belong_A0]+seu[belong_A1];
  seuB = seu[belong_B0]+seu[belong_B1];
  diff_eu = seuB - seuA;
  y ~ bernoulli_logit(tau * diff_eu);
}
generated quantities {
}
