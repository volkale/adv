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
import statsmodels.api as sm

from prepare_data import get_model_input_df

from fit_model import (
    get_varying_intercept_model_results,
    plot_varying_intercept_regression_lines,
    get_shrinkage_plot,
    get_model_results_dict,
    plot_model_comparison_waic,
    plot_model_comparison_CIs,
    plot_posterior_exp_mu,
    get_forest_plot
    )
from lib.summary_stats import stat_funcs, square_stat_funcs, exp_stat_funcs

%matplotlib inline
```

## Investigate linear association between lnMean and lnSD

```python
varying_intercept_model = get_varying_intercept_model_results()

_ = plot_varying_intercept_regression_lines(varying_intercept_model)
_ = get_shrinkage_plot(varying_intercept_model)
```

```python
# naive linear regression
df = get_model_input_df()
sm.OLS(df['lnSD'].values, sm.add_constant(df['lnMean'].values)).fit().summary2()
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
mu_summary = az.summary(
    model_res_dict['remr_lnVR'], stat_funcs=exp_stat_funcs, extend=False, var_names=['mu'], hdi_prob=0.95
)
mu_summary.index = ['$e^\\mu$']
mu_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]
```


```python
beta_summary = az.summary(
    model_res_dict['remr_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['beta'], hdi_prob=0.95
)
beta_summary.index = ['$$\\beta$$']
beta_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]
```

```python
plot_posterior_exp_mu(model_res_dict)
```

```python
beta_summary = az.summary(
    model_res_dict['rema_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['tau'], hdi_prob=0.95
)
beta_summary.index = ['$$\\tau$$']
beta_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]
```

```python
beta_summary = az.summary(
    model_res_dict['rema_lnCVR'], stat_funcs=stat_funcs, extend=False, var_names=['tau'], hdi_prob=0.95
)
beta_summary.index = ['$$\\tau$$']
beta_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]
```

```python
plot_model_comparison_CIs(model_res_dict)
```

```python
data = model_res_dict['remr_lnVR']
get_forest_plot(data)
```

## Prior sensitivity
```python
from sensitivity_analysis import get_prior_comparison

model_res_dict = get_prior_comparison()
dfs = []
for model in ['remr_prior_lnVR_reference', 'remr_prior_lnVR_optimisitc']:
    mu_summary = az.summary(
        model_res_dict[model], stat_funcs=exp_stat_funcs, extend=False, var_names=['mu'], hdi_prob=0.95
    )
    mu_summary.index = [f'{model} $e^\\mu$']
    dfs.append(mu_summary)
pd.concat(dfs)
```
## Baseline severity
```python
from baseline_severity import get_baseline_severity_model, get_baseline_severity_posterior_plot

data = get_baseline_severity_model()
plt = get_baseline_severity_posterior_plot(data)
```

```python
mu_summary = az.summary(
    data, stat_funcs=exp_stat_funcs, extend=False, var_names=['mu'], hdi_prob=0.95
)
mu_summary.index = ['$e^\\mu$']
mu_summary = mu_summary[['Mean' 'Median', 'HDI 2.5%', 'HDI 97.5%']]

tau_summary = az.summary(
    data, stat_funcs=square_stat_funcs, extend=False, var_names=['tau'], hdi_prob=0.95
)
tau_summary.index = ['$$\\tau^2$$']
tau_summary = tau_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]

beta_summary = az.summary(
    data, stat_funcs=stat_funcs, extend=False, var_names=['beta'], hdi_prob=0.95
)
beta_summary.index = ['$\\beta$']
beta_summary = beta_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]


gamma_summary = az.summary(
    data, stat_funcs=stat_funcs, extend=False, var_names=['gamma'], hdi_prob=0.95
)
gamma_summary.index = ['$\gamma$']
gamma_summary = gamma_summary[['Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]


pd.concat(
    [
        mu_summary, tau_summary, beta_summary, gamma_summary
    ]
)
```

## Subgroup analysis
```python
from subgroup_analysis import get_subgroup_models, plot_model_comparison_CIs

model_res_dict = get_subgroup_models()
plt = plot_model_comparison_CIs(model_res_dict)
```

```python
dfs = []
for drug_class in ['atypical', 'ssri', 'ssnri', 'tca']:
    mu_summary = az.summary(
        model_res_dict[drug_class],
        stat_funcs=exp_stat_funcs, extend=False, var_names=['mu'], hdi_prob=0.95, round_to=2
    )
    mu_summary['drug class'] = drug_class
    mu_summary.index = ['$e^\\mu$']
    dfs.append(
        mu_summary[['drug class', 'Mean', 'Median', 'HDI 2.5%', 'HDI 97.5%']]
    )
df_summary = pd.concat(dfs, axis=0)
df_summary
```
## Run simulation
```python
from simulation import get_simulation_results, get_simulation_plots

data = get_simulation_results()
get_simulation_plots(data)
```

```python
np.random.seed(1)
N = 1000
# randomize N patients into active and placebo
W = np.array([False for _ in range(N)])
W[np.random.choice(range(N), int(N / 2), replace=False)] = True

chains = data.posterior.chain.shape[0]
draws = data.posterior.draw.shape[0]
simulations = chains * draws

placebo_sd_response = data.posterior.Ya.values[:, :, :, 0].reshape(simulations, N)[idx, W].std()
active_sd_response = data.posterior.Ya.values[:, :, :, 1].reshape(simulations, N)[idx, ~W].std()

VR = active_sd_response / placebo_sd_response

print(f'VR = {VR:.2f}')

SD_TE = (
    data.posterior.mu.values[:, :, :, 1].reshape(chains * draws, N)
    - data.posterior.mu.values[:, :, :, 0].reshape(chains * draws, N)
)[idx, :].std()

print(f'SD_TE = {SD_TE:.2f}')
```
