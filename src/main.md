---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_model import (
    get_varying_intercept_model_results,
    plot_varying_intercept_regression_lines,
    get_shrinkage_plot,
    get_model_results_dict,
    plot_model_comparison_waic,
    plot_model_comparison_CIs,
    plot_posterior_exp_mu
    )

%matplotlib inline
```

## Investigate linear association between lnMean and lnSD

```python
varying_intervept_model = get_varying_intercept_model_results()

_ = plot_varying_intercept_regression_lines(varying_intervept_model)
_ = get_shrinkage_plot(varying_intervept_model)
```


## Fit models and compare

```python
model_res_dict = get_model_results_dict()
```

```python
az.compare(model_res_dict, scale='log',  seed=1)
```

```python
plot_model_comparison_waic(model_res_dict)
```

```python
stat_funcs = {
    'Mean': lambda x: np.exp(x).mean(),
    'SD': lambda x: np.exp(x).std(),
    'Median': lambda x: np.percentile(np.exp(x), q=[50]),
    'HPD 2.5%': lambda x: az.stats.hpd(np.exp(x), credible_interval=0.95)[0],
    'HPD 97.5%': lambda x: az.stats.hpd(np.exp(x), credible_interval=0.95)[1]
}
mu_summary = az.summary(
    model_res_dict['remr_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['mu'], credible_interval=0.95
)
mu_summary.index = ['$\exp(\mu)$']
mu_summary[['Mean', 'Median', 'HPD 2.5%', 'HPD 97.5%']]
```

```python
columns = [
    ('posterior percentiles', f'{p}th') for p in [5, 25, 50, 75, 95]
]


stat_funcs = {
    ('posterior percentiles', '5th'): lambda x: np.percentile(np.exp(x), q=[5]),
    ('posterior percentiles', '25th'): lambda x: np.percentile(np.exp(x), q=[25]),
    ('posterior percentiles', '50th'): lambda x: np.percentile(np.exp(x), q=[50]),
    ('posterior percentiles', '75th'): lambda x: np.percentile(np.exp(x), q=[75]),
    ('posterior percentiles', '95th'): lambda x: np.percentile(np.exp(x), q=[95])
}
```

```python
mu_summary = az.summary(
    model_res_dict['remr_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['mu'], credible_interval=0.95
)
mu_summary.index = ['REMR $$\exp(\mu)$$']
mu_summary = mu_summary[columns]
mu_summary.columns = pd.MultiIndex.from_tuples(columns)
mu_summary
```

```python
mu_summary = az.summary(
    model_res_dict['rema_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['mu'], credible_interval=0.95
)
mu_summary.index = ['REMA $$\exp(\mu)$$']
mu_summary = mu_summary[columns]
mu_summary.columns = pd.MultiIndex.from_tuples(columns)
mu_summary
```

```python
mu_summary = az.summary(
    model_res_dict['fema_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['mu'], credible_interval=0.95
)
mu_summary.index = ['FEMA $$\exp(\mu)$$']
mu_summary = mu_summary[columns]
mu_summary.columns = pd.MultiIndex.from_tuples(columns)
mu_summary
```

```python
stat_funcs = {
    'Mean': lambda x: x.mean(),
    'Median': lambda x: np.percentile(x, q=[50]),
    '95% HPD l.b.': lambda x: az.stats.hpd(x, credible_interval=0.95)[0],
    '95% HPD u.b.': lambda x: az.stats.hpd(x, credible_interval=0.95)[1]
}
beta_summary = az.summary(
    model_res_dict['remr_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['beta'], credible_interval=0.95
)
beta_summary.index = ['$$\\beta$$']
beta_summary[['Mean', 'Median', '95% HPD l.b.', '95% HPD u.b.']]
```

```python
plot_posterior_exp_mu(model_res_dict)
```

```python
stat_funcs = {
    'Mean': lambda x: (x ** 2).mean(),
    'Median': lambda x: np.percentile(x ** 2, q=[50]),
    'HPD 2.5%': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[0],
    'HPD 97.5%': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[1]
}
beta_summary = az.summary(
    model_res_dict['rema_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['tau'], credible_interval=0.95
)
beta_summary.index = ['$$\\tau^2$$']
beta_summary[['Mean', 'Median', 'HPD 2.5%', 'HPD 97.5%']]
```

```python
stat_funcs = {
    'Mean': lambda x: (x ** 2).mean(),
    'Median': lambda x: np.percentile(x ** 2, q=[50]),
    'HPD 2.5%': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[0],
    'HPD 97.5%': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[1]
}
beta_summary = az.summary(
    model_res_dict['rema_lnCVR'], stat_funcs=stat_funcs, extend=False, var_names=['tau'], credible_interval=0.95
)
beta_summary.index = ['$$\\tau^2$$']
beta_summary[['Mean', 'Median', 'HPD 2.5%', 'HPD 97.5%']]
```

```python
plot_model_comparison_CIs(model_res_dict)
```
