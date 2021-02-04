data {
    int<lower=1> n_y;
    int y[n_y];
    int<lower=1> n_x;
    real x[n_x];
    real p[n_x];
    real tau;
}

transformed data {
}

parameters {
    real<lower=0> rho_u;
    real<lower=0> alpha_u;
    vector<lower=0, upper=1>[n_x] mu_u;
    vector<lower=0, upper=1>[n_x] u;
}

transformed parameters {
    // matrix[n_x, n_x] K_u = cov_exp_quad(x, alpha_u, rho_u) + diag_matrix(rep_vector(1e-9, n_x));
    // matrix[n_x, n_x] K_m = cov_exp_quad(x, alpha_m, rho_m) + diag_matrix(rep_vector(1e-9, n_x));
    // vector[n_x] mu_m = rep_vector(0, n_x);
}

model {
    // vector[n_x] mu_u;
    // vector[n_x] u;
    vector[2] v;
    vector[2] p_choice;
    int c;
    real p_c;

    matrix[n_x, n_x] K_u = cov_exp_quad(x, alpha_u, rho_u) + diag_matrix(rep_vector(1e-9, n_x));

    alpha_u ~ normal(0, 1);
    rho_u ~ normal(0, 1);

    // mu_u ~ multi_normal(mu_m, K_m);
    u ~ multi_normal(mu_u, K_u);

    for (i in 1: n_y) {
        v[#] = p[i] * u[i];
        v[1] = p[n_y+i] * u[n_y+i];
        p_choice = softmax(v ./ tau);
        c = y[i]+1;
        p_c = p_choice[c];
        target += log(p_c);
    }

}
generated quantities {
}