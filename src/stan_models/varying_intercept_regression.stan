data {
    int<lower=1> N; //rows
    real Y_meas[N];
    real X_meas[N];
    real SD_Y[N];
    real SD_X[N];
    int<lower=1> K;
    int<lower=1, upper=K> scale_group[N];
}

parameters {
    real Y[N]; // "true" value of Y
    real X[N]; // "true" value of X
    real alpha[K];  // intercepts
    real beta;  // slope
    real<lower=0> sigma;
}

model {
    sigma ~ cauchy(0, 1);
    alpha ~ normal(0, 1.5);
    beta ~ normal(0, 1);

    Y ~ normal(0, 10);
    X ~ normal(0, 10);

    Y_meas ~ normal(Y, SD_Y);
    X_meas ~ normal(X, SD_X);

    for (i in 1:N){
        Y[i] ~ normal(alpha[scale_group[i]] + beta * X[i], sigma);
    }
}

generated quantities {
    real Y_pred[N];
    real log_lik[N];

    for (i in 1:N) {
        Y_pred[i] = normal_rng(alpha[scale_group[i]] + beta * X[i], sigma);
        log_lik[i] = normal_lpdf(Y[i] | alpha[scale_group[i]] + beta * X[i], sigma);
    }
}
