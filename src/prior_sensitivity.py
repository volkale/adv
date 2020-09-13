import arviz as az
import numpy as np
import os
import pystan
from lib.stan_utils import compile_model
import matplotlib.pyplot as plt

from prepare_data_new import (
    get_formatted_data, add_rank_column, aggregate_treatment_arms, get_variability_effect_sizes
)


# set path to stan model files
dir_name = os.path.dirname(os.path.abspath(__file__))
parent_dir_name = os.path.dirname(dir_name)
stan_model_path = os.path.join(dir_name, 'stan_models')


df = get_formatted_data()
df = df.query('is_placebo_controlled==1')
df = aggregate_treatment_arms(df)
df = get_variability_effect_sizes(df)
for column in ['scale', 'study_id']:
    df = add_rank_column(df, column, ascending=False)


model_res_dict = {}

model = 'remr'
stan_model = compile_model(
    os.path.join(stan_model_path, f'{model}.stan'),
    model_name=model
)
effect_statistic = 'lnVR'

for params in [
    (0., 0.1, 0.1, 0.1),
    (0., 1., 1., 1.),
    (0., 5., 5., 5.),
    (1., 1., 1., 1.),
]:
    prior_mu1, prior_mu2, prior_beta, prior_tau = params
    data_dict = {
        'N': len(df.study_id.unique()),
        'Y_meas': df.groupby(['study_id']).agg({effect_statistic: 'first'}).reset_index()[effect_statistic].values,
        'X_meas': df.groupby(['study_id']).agg({'lnRR': 'first'}).reset_index()['lnRR'].values,
        'SD_Y': np.sqrt(df.groupby(['study_id']).agg(
            {f'var_{effect_statistic}': 'first'}).reset_index()[f'var_{effect_statistic}'].values),
        'SD_X': np.sqrt(df.groupby(['study_id']).agg(
            {'var_lnRR': 'first'}).reset_index()['var_lnRR'].values),
        'mu_prior_loc': 0.,
        'mu_prior_scale': 1.,
        'beta_prior_scale': 1.,
        'tau_prior_scale': 1.,
        'run_estimation': 1
    }

    fit = stan_model.sampling(
        data=data_dict,
        iter=4000,
        warmup=1000,
        chains=3,
        control={'adapt_delta': 0.99},
        check_hmc_diagnostics=True,
        seed=1
    )
    pystan.check_hmc_diagnostics(fit)

    data = az.from_pystan(
        posterior=fit,
        posterior_predictive=['Y_pred'],
        observed_data=['Y_meas', 'X_meas'],
        log_likelihood='log_lik',
    )

    model_res_dict[f'{model}_{effect_statistic}_{params}'] = data

var_names = list(model_res_dict.keys())
data = [
    az.convert_to_dataset({model: np.exp(model_res_dict[model].posterior.mu.values)}) for model in var_names
    ]
_ = az.plot_forest(
    data,
    credible_interval=0.95,
    colors='black',
    figsize=(10, 4),
    var_names=var_names,
    model_names=len(var_names) * ['']
)
plt.title('95% credible intervals for exp(mu) parameter with quartiles')
plt.grid()
