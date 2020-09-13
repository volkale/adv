import arviz as az
import pandas as pd
import numpy as np
import os
import pystan
import matplotlib.pyplot as plt
from lib.stan_utils import compile_model, get_pickle_filename, get_model_code
from prepare_data import get_formatted_data, add_rank_column, aggregate_treatment_arms, get_variability_effect_sizes


# set path to stan model files
dir_name = os.path.dirname(os.path.abspath(__file__))
parent_dir_name = os.path.dirname(dir_name)
stan_model_path = os.path.join(dir_name, 'stan_models')


def get_data_dict(df):
    effect_statistic = 'lnVR'
    return {
        'N': len(df.study_id.unique()),
        'Y_meas': df.groupby(['study_id']).agg({effect_statistic: 'first'}).reset_index()[effect_statistic].values,
        'SD_Y': np.sqrt(df.groupby(['study_id']).agg(
            {f'var_{effect_statistic}': 'first'}).reset_index()[f'var_{effect_statistic}'].values),
        'run_estimation': 1
    }


df = get_formatted_data()

# drug class subgroup analysis
model_res_dict = {}
drug_classes = ['atypical', 'ssri', 'ssnri', 'tca']
for drug_class in drug_classes:
    study_ids = df.query(f'drug_class == "{drug_class}"').study_id.unique()
    df_sub = df[(df.study_id.isin(study_ids)) & (df.drug_class.isin([drug_class, 'placebo']))].copy()
    placebo_controlled_study_ids = set(df_sub.query('is_active == 1')['study_id']) \
        .intersection(df_sub.query('is_active == 0')['study_id'])
    df_sub = df_sub[df_sub.study_id.isin(placebo_controlled_study_ids)]

    for column in ['study_id', 'scale', 'drug_class']:
        df_sub = add_rank_column(df_sub, column)

    df_sub = aggregate_treatment_arms(df_sub)
    df_sub = get_variability_effect_sizes(df_sub)

    model = 'rema'
    stan_model = compile_model(
        os.path.join(stan_model_path, f'{model}.stan'),
        model_name=model
    )

    data_dict = get_data_dict(df_sub)

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

    model_res_dict[drug_class] = data


# plots
def plot_model_comparison_CIs(model_res_dict):
    fig, ax = plt.subplots(nrows=1)
    datasets = [
        az.convert_to_dataset(
            {drug_class: np.exp(model_res_dict[drug_class].posterior.mu.values)}
        ) for drug_class in drug_classes
    ]
    _ = az.plot_forest(
        datasets,
        combined=True,
        credible_interval=0.95,
        quartiles=True,
        colors='black',
        var_names=DRUG_CLASSES,
        model_names=['', '', '', ''],
        ax=ax
    )
    ax.set_title('95% HDI $e^\\mu$')
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir_name, f'output/hdi_drug_class_comparison.tiff'), format='tiff', dpi=500,
                bbox_inches="tight")
    return plt
