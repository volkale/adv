data {
    int<lower=1> N; //rows
    real Y[N];
    real<lower=0> SD_Y[N];
    int<lower=0, upper=1> run_estimation; // a switch to evaluate the likelihood
}

parameters {
    real mu;
}

model {
    mu ~ cauchy(0, 1);

  // likelihood, which we only evaluate conditionally
    if(run_estimation==1){
        for (i in 1:N){
            Y[i] ~ normal(mu, SD_Y[i]);
        }
    }
}

generated quantities {
    real Y_pred[N];
    real log_lik[N];

    for (i in 1:N) {
        Y_pred[i] = normal_rng(mu, SD_Y[i]);
        log_lik[i] = normal_lpdf(Y[i] | mu, SD_Y[i]);
    }
}
