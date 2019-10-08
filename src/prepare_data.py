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


def get_model_input_df(only_placebo_controled=False):
    df = load_data_file()
    df['is_active'] = df['drug'].map(lambda x: x.lower() != 'placebo').astype(int)
    # perform type conversions
    df['mean_endpoint'] = df['mean_endpoint'].map(lambda x: float(x) if x != '*' else None)
    df['sd_endpoint'] = df['sd_endpoint'].map(lambda x: float(x) if x != '*' else None)
    df['no_randomised'] = df['no_randomised'].map(lambda x: int(x) if x != '*' else None)

    # filter out studies in which the mean does not represent a change
    df = df[df['mean_endpoint'] < 0]

    # rename column to get positive numbers
    df['negative_change_mean'] = - df['mean_endpoint']
    df['negative_change_sd'] = df['sd_endpoint']

    # aggregate treatment arms from the same study, as advised in:
    # https://handbook-5-1.cochrane.org/chapter_16/16_5_4_how_to_include_multiple_groups_from_one_study.htm
    df_agg = get_pooled_data(
        df,
        arm_size='no_randomised',
        mean='negative_change_mean',
        sd='negative_change_sd'
    )
    df_agg['CV'] = df_agg['negative_change_sd'] / df_agg['negative_change_mean']
    df_agg['lnCV'] = np.log(df_agg['CV'])

    if only_placebo_controled:
        # only choose study ids that have placebo and active arm
        placebo_controlled_study_ids = set(df_agg.query('is_active == 1')['study_id']) \
            .intersection(df_agg.query('is_active == 0')['study_id'])

        df_agg = df_agg[df_agg.study_id.isin(placebo_controlled_study_ids)].copy()

        # compute variability metrics
        get_var_lnCVR = var_lnCVR_factory(
            mean='negative_change_mean',
            sd='negative_change_sd'
        )

        lnRR = df_agg.groupby(['study_id']).apply(get_lnRR).reset_index().rename(columns={0: 'lnRR'})
        var_lnRR = df_agg.groupby(['study_id']).apply(get_var_lnRR).reset_index().rename(columns={0: 'var_lnRR'})
        lnVR = df_agg.groupby(['study_id']).apply(get_lnVR).reset_index().rename(columns={0: 'lnVR'})
        var_lnVR = df_agg.groupby(['study_id']).apply(get_var_lnVR).reset_index().rename(columns={0: 'var_lnVR'})
        lnCVR = df_agg.groupby(['study_id']).apply(get_lnCVR).reset_index().rename(columns={0: 'lnCVR'})
        var_lnCVR = df_agg.groupby(['study_id']).apply(get_var_lnCVR).reset_index().rename(columns={0: 'var_lnCVR'})

        df_agg = df_agg \
            .merge(lnRR, on=['study_id']) \
            .merge(var_lnRR, on=['study_id']) \
            .merge(lnVR, on=['study_id']) \
            .merge(var_lnVR, on=['study_id']) \
            .merge(lnCVR, on=['study_id']) \
            .merge(var_lnCVR, on=['study_id'])

    df_agg['lnMean'] = df_agg.apply(lambda x: np.log(x['negative_change_mean']), axis=1)
    df_agg['lnSD'] = df_agg.apply(lambda x: get_lnSD(x['negative_change_sd'], x['N']), axis=1)
    df_agg['var_lnMean'] = df_agg.apply(
        lambda x: get_var_lnMean(x['negative_change_mean'], x['negative_change_sd'], x['N']),
        axis=1
    )
    df_agg['var_lnSD'] = df_agg.apply(lambda x: get_var_lnSD(x['N']), axis=1)

    df_agg['study_rank'] = df_agg['study_id'].rank(method='dense').astype(int)
    df_agg['scale_rank'] = df_agg['scale'].rank(method='dense').astype(int)

    return df_agg
