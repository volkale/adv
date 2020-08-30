import pandas as pd
import numpy as np


def get_pooled_mean(n_i: np.array, mu_i: np.array) -> np.array:
    return np.sum(n_i * mu_i) / np.sum(n_i)


def get_pooled_sd(n_i: np.array, mu_i: np.array, sd_i: np.array) -> np.array:
    # TODO: check this formula, source: wikipedia.org
    return np.sqrt(
        (
            np.sum((n_i - 1) * sd_i ** 2 + n_i * mu_i ** 2) - np.sum(n_i) * get_pooled_mean(n_i, mu_i) ** 2
        ) / np.sum(n_i - 1)
    )


def get_pooled_data(df, arm_size, mean, sd=None):
    df_agg = pd.DataFrame(
        df.groupby(['study_id', 'is_active']).apply(
            lambda x: pd.DataFrame(
                [
                    np.sum(x[arm_size]),
                    get_pooled_mean(x[arm_size], x[mean]),
                    get_pooled_sd(x[arm_size], x[mean], x[sd]) if sd else 0,
                    '-'.join(np.unique(x['scale']))
                ]
            ).T
        )
    ).reset_index()
    if sd:
        df_agg = df_agg.rename(columns={0: 'N', 1: mean, 2: sd, 3: 'scale'}).astype({'N': int, mean: float, sd: float})
    else:
        df_agg = df_agg.rename(columns={0: 'N', 1: mean, 2: 'scale'}).astype({'N': int, mean: float})
    del df_agg['level_2']
    return df_agg
