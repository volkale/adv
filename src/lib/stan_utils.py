import logging
import pandas as pd
import pickle
import pystan

logging.basicConfig(level=logging.INFO, filename='info.log', filemode='w')


def compile_model(filename, model_name=None, **kwargs):
    """This will automatically cache models - great if you're just running a
    script on the command line.
    See http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html"""
    cache_fn = get_pickle_filename(filename, model_name=model_name)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except FileNotFoundError:
        model_code = get_model_code(filename)
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logging.info("Using cached StanModel")
    return sm


def get_pickle_filename(filename, model_name=None):
    from hashlib import md5
    with open(filename) as f:
        model_code = f.read()
        code_hash = md5(model_code.encode('ascii')).hexdigest()
        import os
        file_path = os.path.dirname(filename)
        if model_name is None:
            cache_fn = 'cached-model-{}.pkl'.format(code_hash)
        else:
            cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    return os.path.join(file_path, cache_fn)


def get_model_code(filename):
    with open(filename) as f:
        model_code = f.read()
    return model_code


def get_stan_df(fit):
    summary_dict = fit.summary()
    return (
        pd.DataFrame(
            summary_dict['summary'],
            columns=summary_dict['summary_colnames'],
            index=summary_dict['summary_rownames']
        )
    )
