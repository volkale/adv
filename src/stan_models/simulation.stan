data {
    int<lower=0> N;                // number of simulated individuals
    real<lower=-1, upper=1> rho;   // assumed correlation between response under placebo and individual treatment effect
    real<lower=0> sd_te;           // SD of individual treatment effect
    real<lower=0> sd_m;            // measurement error of outcome variable
    real<lower=0, upper=1> lambda; // mixture weight, fraction of (positive) responders
    real<lower=0, upper=1> theta;  // the variance of the mixture is the mixture of the variances plus a non-negative
                                   // term accounting for the (weighted) dispersion of the means,
                                   // this parameter controls how much of the variation comes from the ladder part
}

parameters {
    matrix[2, N] z_u;
}

transformed parameters {
    real tau;                      // average response
    real delta;                    // ATE
    vector<lower=0>[2] sigma_u;    // SD of response under placebo and SD of individual treatment effect

    real mu_high;
    real mu_low;
    real<lower=0> sigma;
    
    real mu[N, 2];
    matrix[2, N] u;
    cholesky_factor_cov[2, 2] L_u;

    if(lambda < 1 && lambda > 0) {
        mu_high = theta * sqrt((1. - lambda) / lambda);
        mu_low  = - lambda * mu_high / (1. - lambda);
        sigma = sqrt(1. - lambda * square(mu_high) - (1. - lambda) * square(mu_low));
    }
    else {
        mu_high = 0;
        mu_low = 0.;
        sigma = 1.;
    }

    tau = 8.8;                     // Cipriani et al. data
    delta = 2;

    L_u[1, 1] = 1;
    L_u[2, 2] = 1;
    L_u[2, 1] = rho;
    L_u[1, 2] = 0;
    
    sigma_u[1] = 7.7;              // Cipriani et al. data SD of response 7.7
    sigma_u[2] = sd_te;

    u = diag_pre_multiply(sigma_u, L_u) * z_u;

    for (i in 1:N) {
        for (a in 1:2) {
            mu[i, a] = tau + u[1, i] + (a - 1) * (delta + u[2, i]);
        }
    }
}

model {
    if(lambda < 1 && lambda > 0) {
        for (n in 1:N) {
            target += normal_lpdf(z_u[1, n] | 0, 1);
            target += log_mix(
                lambda, 
                normal_lpdf(z_u[2, n] | mu_high, sigma),
                normal_lpdf(z_u[2, n] | mu_low, sigma)
                );
        }
    }
    else {
        z_u[2] ~ normal(0, 1);
    }
}

generated quantities {
    real Ya[N, 2];

    for (i in 1:N) {
        Ya[i, 1] = normal_rng(mu[i, 1], sd_m);  // potential outcome under placebo
        Ya[i, 2] = normal_rng(mu[i, 2], sd_m);  // potential outcome under active
        }
}
