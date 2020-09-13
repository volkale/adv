import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from lib.drug_classes import get_drug_class
from lib.max_values import MAX_VALUES
from lib.pool_arms import get_pooled_data, get_pooled_mean, get_pooled_sd
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
    used_columns = ['study_id', 'drug', 'no_randomised', 'mean_endpoint', 'sd_endpoint', 'scale', 'mean_pre', 'weeks']
    return pd.DataFrame(df.iloc[2:, :].values, columns=all_columns)[used_columns]


def get_formatted_data():
    df = load_data_file()
    # perform type conversions
    for c in ['mean_pre', 'mean_endpoint', 'sd_endpoint', 'no_randomised']:
        df[c] = df[c].map(lambda x: float(x) if x != '*' else None)
    df['weeks'] = df.weeks.map(lambda x: int(x) if x != '*' else None)
    # filter out studies in which the mean does not represent a change
    df = df[df['mean_endpoint'] < 0]
    # corrected corrupted data, cf. https://pubmed.ncbi.nlm.nih.gov/9733503
    df['scale'] = df.apply(
        lambda x: "HAMD21" if x['study_id'] == "Kramer1998" else x['scale'],
        axis=1
    )
    # maximum values for scales
    df['max_scale_value'] = df.scale.map(lambda x: MAX_VALUES.get(x, 0.))
    # add flags
    df['is_active'] = df['drug'].map(lambda x: x.lower() != 'placebo').astype(int)
    # studies for which at least one arm has mean pre
    study_ids_with_mean_pre = list(
        df.groupby('study_id').agg({'mean_pre': lambda x: int(True in set(x > 0))}).query('mean_pre==1').index.unique()
    )
    df['has_mean_pre'] = df.study_id.isin(study_ids_with_mean_pre).astype(int)
    # studies with placebo arm
    placebo_controlled_study_ids = set(df.query('is_active == 1')['study_id']) \
        .intersection(df.query('is_active == 0')['study_id'])
    df['is_placebo_controlled'] = df.study_id.isin(placebo_controlled_study_ids).astype(int)
    df['known_max_value'] = df['max_scale_value'].map(lambda x: int(x > 0.))
    # rename column to get positive numbers
    df['negative_change_mean'] = - df['mean_endpoint']
    df['negative_change_sd'] = df['sd_endpoint']
    del df['mean_endpoint']
    del df['sd_endpoint']
    # rename number of patients column
    df.rename(columns={'no_randomised': 'N'}, inplace=True)
    df['N'] = df['N'].astype(int)
    # add drug class
    df['drug_class'] = df.drug.map(get_drug_class)
    # add log values for Mean and SD, and their respective variance
    df['lnMean'] = df.apply(lambda x: np.log(x['negative_change_mean']), axis=1)
    df['lnSD'] = df.apply(lambda x: get_lnSD(x['negative_change_sd'], x['N']), axis=1)
    df['var_lnMean'] = df.apply(
        lambda x: get_var_lnMean(x['negative_change_mean'], x['negative_change_sd'], x['N']),
        axis=1
    )
    df['var_lnSD'] = df.apply(lambda x: get_var_lnSD(x['N']), axis=1)
    return df


def get_rescaled_data(df):
    df_norm = df.copy()
    for column in ['negative_change_mean', 'negative_change_sd', 'mean_pre']:
        df_norm[column] = df_norm.apply(
            lambda x: x[column] / MAX_VALUES[x['scale']] if MAX_VALUES.get(x['scale']) else None,
            axis=1
        )
    return df_norm


def aggregate_treatment_arms(df):
    # aggregate treatment arms from the same study, as advised in:
    # https://handbook-5-1.cochrane.org/chapter_16/16_5_4_how_to_include_multiple_groups_from_one_study.htm
    df_agg = pd.DataFrame(
        df.groupby(['study_id', 'is_active']).apply(
            lambda x: pd.DataFrame(
                [
                    np.sum(x['N']),
                    get_pooled_mean(x['N'], x['mean_pre']),
                    get_pooled_mean(x['N'], x['negative_change_mean']),
                    get_pooled_sd(x['N'], x['negative_change_mean'], x['negative_change_sd']),
                    '-'.join(np.unique(x['scale'])),
                    np.max(x['max_scale_value']),
                    get_pooled_mean(x['N'], x['weeks'])
                ]
            ).T
        )
    ).reset_index()
    df_agg = df_agg.rename(
        columns={
            0: 'N', 1: 'mean_pre', 2: 'negative_change_mean', 3: 'negative_change_sd', 4: 'scale',
            5: 'max_scale_value', 6: 'weeks'
        }
    ).astype({'N': int, 'negative_change_mean': float, 'negative_change_sd': float})
    del df_agg['level_2']

    df_agg['CV'] = df_agg['negative_change_sd'] / df_agg['negative_change_mean']
    df_agg['lnCV'] = np.log(df_agg['CV'])
    return df_agg


def get_variability_effect_sizes(df):
    assert df.groupby('study_id').agg(
        {'is_active': lambda x: set(x) == {0, 1}}
    ).is_active.all()
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
    baseline = df.groupby(['study_id']).apply(
        lambda x: np.sum(x['mean_pre'] * x['N'] / x['max_scale_value']) / np.sum(x['N'])
    ).reset_index().rename(columns={0: 'baseline'})

    df = df \
        .merge(lnRR, on=['study_id']) \
        .merge(var_lnRR, on=['study_id']) \
        .merge(lnVR, on=['study_id']) \
        .merge(var_lnVR, on=['study_id']) \
        .merge(lnCVR, on=['study_id']) \
        .merge(var_lnCVR, on=['study_id']) \
        .merge(baseline, on=['study_id'])

    df['lnMean'] = df.apply(lambda x: np.log(x['negative_change_mean']), axis=1)
    df['lnSD'] = df.apply(lambda x: get_lnSD(x['negative_change_sd'], x['N']), axis=1)
    df['var_lnMean'] = df.apply(
        lambda x: get_var_lnMean(x['negative_change_mean'], x['negative_change_sd'], x['N']),
        axis=1
    )
    df['var_lnSD'] = df.apply(lambda x: get_var_lnSD(x['N']), axis=1)
    return df


def get_model_input_df():
    df = get_formatted_data()
    df = df.query('is_placebo_controlled==1')
    df = aggregate_treatment_arms(df)
    df = get_variability_effect_sizes(df)
    df = add_rank_column(df, 'study_id', ascending=False)
    df = add_rank_column(df, 'scale', ascending=False)
    return df


def add_rank_column(dataframe, column, ascending=False):
    dataframe[f'{column}_rank'] = dataframe[column].rank(method='dense', ascending=ascending).astype(int)
    return dataframe
