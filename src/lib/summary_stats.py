import arviz as az
import numpy as np


stat_funcs = {
    'Mean': lambda x: x.mean(),
    'Median': lambda x: np.percentile(x, q=[50]),
    'HDI 2.5%': lambda x: az.stats.hdi(x, hdi_prob=0.95)[0],
    'HDI 97.5%': lambda x: az.stats.hdi(x, hdi_prob=0.95)[1]
}

square_stat_funcs = {
    'Mean': lambda x: (x ** 2).mean(),
    'Median': lambda x: np.percentile(x ** 2, q=[50]),
    'HDI 2.5%': lambda x: az.stats.hdi(x ** 2, hdi_prob=0.95)[0],
    'HDI 97.5%': lambda x: az.stats.hdi(x ** 2, hdi_prob=0.95)[1]
}

exp_stat_funcs = {
    'Mean': lambda x: np.exp(x).mean(),
    'SD': lambda x: np.exp(x).std(),
    'Median': lambda x: np.percentile(np.exp(x), q=[50]),
    'HDI 2.5%': lambda x: az.stats.hdi(np.exp(x), hdi_prob=0.95)[0],
    'HDI 97.5%': lambda x: az.stats.hdi(np.exp(x), hdi_prob=0.95)[1]
}
