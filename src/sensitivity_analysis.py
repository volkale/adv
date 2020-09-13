import arviz as az
import numpy as np
import os
import pystan
from prepare_data import get_model_input_df

from lib.stan_utils import compile_model

# set path to stan model files
dir_name = os.path.dirname(os.path.abspath(__file__))
parent_dir_name = os.path.dirname(dir_name)
stan_model_path = os.path.join(dir_name, 'stan_models')
# investigate correlation between lnMean and lnSD


def get_prior_comparison():
    df = get_model_input_df()
    model_res_dict = {}

    model = 'remr_prior'
    stan_model = compile_model(
        os.path.join(stan_model_path, f'{model}.stan'),
        model_name=model
    )
    effect_statistic = 'lnVR'
    data_dict = get_data_dict(df, effect_statistic)

    prior_dict = {
        'reference': (0, 1),
        'optimisitc': (np.log(2), 0.43)
    }
    # from scipy import stats
    # stats.norm.cdf(0, loc=np.log(2), scale=0.43)
    # 0.05348421366569122
    for prior, (mu_prior_loc, mu_prior_scale) in prior_dict.items():
        data_dict_prior = data_dict.copy()

        data_dict_prior['mu_prior_loc'] = mu_prior_loc
        data_dict_prior['mu_prior_scale'] = mu_prior_scale

        fit = stan_model.sampling(
            data=data_dict_prior,
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
        model_res_dict[f'{model}_{effect_statistic}_{prior}'] = data
    return model_res_dict


def get_data_dict(df, effect_statistic):
    return {
        'N': len(df.study_id.unique()),
        'Y': df.groupby(['study_id']).agg({effect_statistic: 'first'}).reset_index()[effect_statistic].values,
        'Y_meas': df.groupby(['study_id']).agg({effect_statistic: 'first'}).reset_index()[effect_statistic].values,
        'X_meas': df.groupby(['study_id']).agg({'lnRR': 'first'}).reset_index()['lnRR'].values,
        'SD_Y': np.sqrt(df.groupby(['study_id']).agg(
            {f'var_{effect_statistic}': 'first'}).reset_index()[f'var_{effect_statistic}'].values),
        'SD_X': np.sqrt(df.groupby(['study_id']).agg(
            {'var_lnRR': 'first'}).reset_index()['var_lnRR'].values),
        'run_estimation': 1
    }
