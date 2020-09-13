import arviz as az
import numpy as np
import os
import pystan
import matplotlib.pyplot as plt
from lib.stan_utils import compile_model

# set path to stan model files
dir_name = os.path.dirname(os.path.abspath(__file__))
parent_dir_name = os.path.dirname(dir_name)
stan_model_path = os.path.join(dir_name, 'stan_models')


def get_simulation_results():
    data_dict = {
        'N': 1000,
        'rho': -0.4,
        'sd_te': 6.5,
        'sd_m': 0.001,
        'lambda': 0.2,
        'theta': 0.9
    }
    simulation_stan_model = compile_model(
        os.path.join(stan_model_path, 'simulation.stan'),
        model_name='simulation'
    )
    fit = simulation_stan_model.sampling(
        data=data_dict,
        warmup=500,
        iter=2500,
        chains=2,
        check_hmc_diagnostics=True,
        seed=1
    )
    pystan.check_hmc_diagnostics(fit)
    data = az.from_pystan(posterior=fit)
    return data


def get_simulation_plots(data):
    chains = data.posterior.chain.shape[0]
    draws = data.posterior.draw.shape[0]
    simulations = chains * draws
    N = 1000  # number of patients in the simulation
    idx = 1000  # pick one simulated data set

    placebo_response = data.posterior.mu.values[:, :, :, 0].reshape(simulations, N)[idx, :]
    active_response = data.posterior.mu.values[:, :, :, 1].reshape(simulations, N)[idx, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Histogram of potential outcome response under placebo and active treatment')

    _ = ax.hist(placebo_response, bins=35, color='blue', histtype='step', label='placebo')
    _ = ax.hist(active_response, bins=35, color='red', histtype='step', label='active')
    ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig(os.path.join(parent_dir_name, f'output/hist_po.tiff'), format='tiff', dpi=500, bbox_inches="tight")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex='col', sharey='col')
    fig.suptitle('Potential outcome responses with baseline gauged to 0')

    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(['pre', 'post'])
    for pr in placebo_response:
        _ = axes[0, 0].plot([0, 1], [0, pr], linestyle='-', alpha=0.1, color='blue', label='placebo')
    axes[0, 0].set_ylabel('response in HAMD17')

    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(['pre', 'post'])
    for ar in active_response:
        _ = axes[1, 0].plot([0, 1], [0, ar], linestyle='-', alpha=0.1, color='red', label='active')
    axes[1, 0].set_ylabel('response in HAMD17')

    _ = axes[0, 1].hist(placebo_response, orientation="horizontal", color='blue', label='placebo', bins=35,
                        histtype='step', density=True)
    axes[0, 1].set_xticks([])
    axes[0, 1].legend()
    _ = axes[1, 1].hist(active_response, orientation="horizontal", color='red', label='active', bins=35,
                        histtype='step', density=True)
    axes[1, 1].set_xticks([])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig(os.path.join(parent_dir_name, f'output/po_reponse.tiff'), format='tiff', dpi=500, bbox_inches="tight")

    np.random.seed(0)
    # randomize N patients into active and placebo
    W = np.array([False for _ in range(N)])
    W[np.random.choice(range(N), int(N / 2), replace=False)] = True

    placebo_response = data.posterior.Ya.values[:, :, :, 0].reshape(chains * draws, N)[idx, W]
    active_response = data.posterior.Ya.values[:, :, :, 1].reshape(chains * draws, N)[idx, W]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    fig.suptitle(
        'Individual treatment effect visualized as slope \n blue if active produces larger response, red otherwise')

    plt.xticks([0, 1], ['placebo', 'active'])
    for pr, ar in zip(placebo_response, active_response):
        _ = axes.plot([0, 1], [pr, ar], linestyle='-', alpha=0.1, color='blue' if ar > pr else 'red')

    axes.set_ylabel('response in HAMD17')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig(os.path.join(parent_dir_name, f'output/indiv_te.tiff'), format='tiff', dpi=500, bbox_inches="tight")

    return plt
