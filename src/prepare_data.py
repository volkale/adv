import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from lib.pool_arms import get_pooled_data
from lib.variability_metrics import get_lnVR
from lib.variability_metrics import get_var_lnVR
from lib.variability_metrics import get_lnRR
from lib.variability_metrics import get_var_lnRR
from lib.variability_metrics import get_lnCVR
from lib.variability_metrics import var_lnCVR_factory
from lib.variability_metrics import get_lnSD
from lib.variability_metrics import get_var_lnSD
from lib.variability_metrics import get_var_lnMean


plt.style.use('ggplot')


parent_dir_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data_file():
    df = pd.read_excel(os.path.join(parent_dir_name, 'data/Cipriani et al_GRISELDA_Lancet 2018_Open data.xlsx'))
    all_columns = [
        'study_id', 'year_published', 'drug', 'no_randomised', 'responders',
        'is_imputed_baseline', 'remitters', 'is_imputed_endpoint', 'N comp+imputed_baseline',
        'definition_of_response', 'definition_of_remission', 'dropouts_total',
        'dropouts_sideeffects', 'scale', 'mean_pre', 'weeks', 'N comp+imputed_endpoint',
        'mean_endpoint', 'sd_endpoint'
    ]
    used_columns = ['study_id', 'drug', 'no_randomised', 'mean_endpoint', 'sd_endpoint', 'scale']
    return pd.DataFrame(df.iloc[2:, :].values, columns=all_columns)[used_columns]


def get_model_input_df(pool_arms=True, only_placebo_controled=True):
    df = load_data_file()

    df = prepare_data(df)

    if pool_arms:
        df = get_pooled_data(df, arm_size='N', mean='negative_change_mean', sd='negative_change_sd')

    df['CV'] = df['negative_change_sd'] / df['negative_change_mean']
    df['lnCV'] = np.log(df['CV'])

    if only_placebo_controled:
        # only choose study ids that have placebo and active arm
        placebo_controlled_study_ids = set(df.query('is_active == 1')['study_id']) \
            .intersection(df.query('is_active == 0')['study_id'])

        df = df[df.study_id.isin(placebo_controlled_study_ids)].copy()

        # compute variability metrics
        get_var_lnCVR = var_lnCVR_factory(
            mean='negative_change_mean',
            sd='negative_change_sd'
        )

        lnRR = df.groupby(['study_id']).apply(get_lnRR).reset_index().rename(columns={0: 'lnRR'})
        var_lnRR = df.groupby(['study_id']).apply(get_var_lnRR).reset_index().rename(columns={0: 'var_lnRR'})
        lnVR = df.groupby(['study_id']).apply(get_lnVR).reset_index().rename(columns={0: 'lnVR'})
        var_lnVR = df.groupby(['study_id']).apply(get_var_lnVR).reset_index().rename(columns={0: 'var_lnVR'})
        lnCVR = df.groupby(['study_id']).apply(get_lnCVR).reset_index().rename(columns={0: 'lnCVR'})
        var_lnCVR = df.groupby(['study_id']).apply(get_var_lnCVR).reset_index().rename(columns={0: 'var_lnCVR'})

        df = df \
            .merge(lnRR, on=['study_id']) \
            .merge(var_lnRR, on=['study_id']) \
            .merge(lnVR, on=['study_id']) \
            .merge(var_lnVR, on=['study_id']) \
            .merge(lnCVR, on=['study_id']) \
            .merge(var_lnCVR, on=['study_id'])

    df['lnMean'] = df.apply(lambda x: np.log(x['negative_change_mean']), axis=1)
    df['lnSD'] = df.apply(lambda x: get_lnSD(x['negative_change_sd'], x['N']), axis=1)
    df['var_lnMean'] = df.apply(
        lambda x: get_var_lnMean(x['negative_change_mean'], x['negative_change_sd'], x['N']),
        axis=1
    )
    df['var_lnSD'] = df.apply(lambda x: get_var_lnSD(x['N']), axis=1)

    df['study_rank'] = df['study_id'].rank(method='dense').astype(int)
    df['scale_rank'] = df['scale'].rank(method='dense').astype(int)

    return df


def prepare_data(df):
    df = convert_types(df)
    df = filter_studies(df)
    df = generate_new_columns(df)
    return df


def generate_new_columns(df):
    df['is_active'] = df['drug'].map(lambda x: x.lower() != 'placebo').astype(int)
    # rename column to get positive numbers
    df['negative_change_mean'] = - df['mean_endpoint']
    df['negative_change_sd'] = df['sd_endpoint']
    df.rename(columns={'no_randomised': 'N'}, inplace=True)
    return df


def filter_studies(df):
    # filter out studies in which the mean does not represent a change
    df = df[df['mean_endpoint'] < 0]
    return df


def convert_types(df):
    df['mean_endpoint'] = df['mean_endpoint'].map(lambda x: float(x) if x != '*' else None)
    df['sd_endpoint'] = df['sd_endpoint'].map(lambda x: float(x) if x != '*' else None)
    df['no_randomised'] = df['no_randomised'].map(lambda x: int(x) if x != '*' else None)
    return df
