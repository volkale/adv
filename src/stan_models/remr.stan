data {
    int<lower=1> N; //rows
    real Y_meas[N];
    real X_meas[N];
    real<lower=0> SD_Y[N];
    real<lower=0> SD_X[N];
    int<lower=0, upper=1> run_estimation; // a switch to evaluate the likelihood
}

parameters {
    real mu;
    real beta;
    real<lower=0> tau; //between study variance
    real eta[N];

    real X[N];
}

transformed parameters {
  real Y[N];
  for (i in 1:N)
    Y[i] = mu + beta * X[i] + tau * eta[i];
}

model {
    mu ~ cauchy(0, 1);
    beta ~ cauchy(0, 1);
    tau ~ cauchy(0, 1);
    eta ~ normal(0, 1);

    X ~ normal(0, 2.5);
    X_meas ~ normal(X, SD_X);

  // likelihood, which we only evaluate conditionally
    if(run_estimation==1){
        for (i in 1:N){
            Y_meas[i] ~ normal(Y[i], SD_Y[i]);
        }
    }
}

generated quantities {
    real Y_pred[N];
    real log_lik[N];

    for (i in 1:N) {
        Y_pred[i] = normal_rng(mu + beta * X[i], tau);
        log_lik[i] = normal_lpdf(Y_meas[i] | Y[i], SD_Y[i]);
    }
}
