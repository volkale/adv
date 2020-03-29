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
varying_intercept_model = get_varying_intercept_model_results()

_ = plot_varying_intercept_regression_lines(varying_intercept_model)
_ = get_shrinkage_plot(varying_intercept_model)
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
    '95% HPD l.b.': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[0],
    '95% HPD u.b.': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[1]
}
beta_summary = az.summary(
    model_res_dict['rema_lnVR'], stat_funcs=stat_funcs, extend=False, var_names=['tau'], credible_interval=0.95
)
beta_summary.index = ['$$\\tau^2$$']
beta_summary[['Mean', 'Median', '95% HPD l.b.', '95% HPD u.b.']]
```

```python
stat_funcs = {
    'Mean': lambda x: (x ** 2).mean(),
    'Median': lambda x: np.percentile(x ** 2, q=[50]),
    '95% HPD l.b.': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[0],
    '95% HPD u.b.': lambda x: az.stats.hpd(x ** 2, credible_interval=0.95)[1]
}
beta_summary = az.summary(
    model_res_dict['rema_lnCVR'], stat_funcs=stat_funcs, extend=False, var_names=['tau'], credible_interval=0.95
)
beta_summary.index = ['$$\\tau^2$$']
beta_summary[['Mean', 'Median', '95% HPD l.b.', '95% HPD u.b.']]
```

```python
plot_model_comparison_CIs(model_res_dict)
```


## Run simulation
```python
from simulation import get_simulation_results

data = get_simulation_results()
```

```python
chains = data.posterior.chain.shape[0]
draws = data.posterior.draw.shape[0]
simulations = chains * draws
N = 1000  # number of patients in the simulation
idx = 750  # pick one simulated data set

placebo_response = data.posterior.mu.values[:, :, :, 0].reshape(simulations, N)[idx, :]
active_response = data.posterior.mu.values[:, :, :, 1].reshape(simulations, N)[idx, :]

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Histogram of potential outcome response under placebo and active treatment')

_ = ax.hist(placebo_response, bins=35, color='blue', histtype='step', label='placebo')
_ = ax.hist(active_response, bins=35, color='red',  histtype='step', label='active')
ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex='col', sharey='col')
fig.suptitle('Potential outcome responses with baseline gauged to 0')

for pr in placebo_response:
    _ = axes[0, 0].plot([0, 1], [0, pr], linestyle='-', alpha=0.1, color='blue', label='placebo')
    axes[0, 0].set_ylabel('response in HAMD17')
_ = axes[0, 1].hist(placebo_response, orientation="horizontal", color='blue', label='placebo', bins=35, histtype='step', density=True)
axes[0, 1].legend()

for ar in active_response:
    _ = axes[1, 0].plot([0, 1], [0, ar], linestyle='-', alpha=0.1, color='red', label='active')
axes[1, 0].set_ylabel('response in HAMD17')
_ = axes[1, 1].hist(active_response, orientation="horizontal", color='red', label='active', bins=35, histtype='step', density=True)
axes[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1)


np.random.seed(0)
W = np.random.binomial(1, 0.8, N) == 0  # randomize N patients into active and placebo

placebo_response = data.posterior.Ya.values[:, :, :, 0].reshape(chains * draws, N)[idx, W]
active_response = data.posterior.Ya.values[:, :, :, 1].reshape(chains * draws, N)[idx, W]

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
fig.suptitle('Individual treatment effect visualized as slope \n blue if active produces larger response, red otherwise')

for pr, ar in zip(placebo_response, active_response):
    _ = axes.plot([0, 1], [pr, ar], linestyle='-', alpha=0.1, color='blue' if ar > pr else 'red')

axes.set_ylabel('response in HAMD17')

plt.tight_layout()
plt.subplots_adjust(top=0.9, bottom=0.1)

```

```python
np.random.seed(0)
W = np.random.binomial(1, 0.5, N) <= 0.5  # randomize N patients into active and placebo

placebo_sd_response = data.posterior.Ya.values[:, :, :, 0].reshape(simulations, N)[idx, W].std()
active_sd_response = data.posterior.Ya.values[:, :, :, 1].reshape(simulations, N)[idx, ~W].std()

VR = active_sd_response / placebo_sd_response  # + 1 / (2 * (np.sum(W) - 1)) - 1 / (2 * (np.sum(~W) - 1))

print(f'VR = {VR:.2f}')  # ignoring the small sample correction terms

SD_TE = (
    data.posterior.mu.values[:, :, :, 1].reshape(chains * draws, N)
    - data.posterior.Ya.values[:, :, :, 0].reshape(chains * draws, N)
)[idx, :].std()

print(f'SD_TE = {SD_TE:.2f}')
```
