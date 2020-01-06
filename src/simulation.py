import arviz as az
import os
import pystan

from lib.stan_utils import compile_model

# set path to stan model files
dir_name = os.path.dirname(os.path.abspath(__file__))
parent_dir_name = os.path.dirname(dir_name)
stan_model_path = os.path.join(dir_name, 'stan_models')


def get_simulation_results():
    data_dict = {
        'N': 1000,
        'rho': -0.6,
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
