import numpy as np

# NOTE
# https://stats.stackexchange.com/questions/57715/expected-value-and-variance-of-loga


def get_lnRR(x):
    X_a = np.sum(x['negative_change_mean'] * x['is_active'])
    X_p = np.sum(x['negative_change_mean']) - X_a
    return np.log(X_a / X_p)


def get_var_lnRR(x):
    CV_a = np.sum(x['CV'] * x['is_active'])
    CV_p = np.sum(x['CV']) - CV_a
    N_a = np.sum(x['N'] * x['is_active'])
    N_p = np.sum(x['N']) - N_a
    return CV_a ** 2 / N_a + CV_p ** 2 / N_p


def get_lnVR(x):
    SD_a = np.sum(x['negative_change_sd'] * x['is_active'])
    SD_p = np.sum(x['negative_change_sd']) - SD_a
    N_a = np.sum(x['N'] * x['is_active'])
    N_p = np.sum(x['N']) - N_a
    return np.log(SD_a / SD_p) + 1 / 2 / (N_a - 1) - 1 / 2 / (N_p - 1)


def get_var_lnVR(x):
    N_a = np.sum(x['N'] * x['is_active'])
    N_p = np.sum(x['N']) - N_a
    return 1 / 2 / (N_a - 1) + 1 / 2 / (N_p - 1)


def get_lnCVR(x):
    CV_a = np.sum(x['CV'] * x['is_active'])
    CV_p = np.sum(x['CV']) - CV_a
    N_a = np.sum(x['N'] * x['is_active'])
    N_p = np.sum(x['N']) - N_a
    return np.log(CV_a / CV_p) + 1 / 2 / (N_a - 1) - 1 / 2 / (N_p - 1)


def var_lnCVR_factory(mean, sd):

    def get_var_lnCVR(x):
        N_a = np.sum(x['N'] * x['is_active'])
        N_p = np.sum(x['N']) - N_a
        mean_a = np.sum(x[mean] * x['is_active'])
        mean_p = np.sum(x[mean]) - mean_a
        samp_std_a = np.sum(x[sd] * x['is_active'])
        samp_std_p = np.sum(x[sd]) - samp_std_a
        return (
            _util_func(N_a, mean_a, samp_std_a) +
            _util_func(N_p, mean_p, samp_std_p)
        )

    return get_var_lnCVR


def _util_func(N, mean, samp_std):
    a = samp_std ** 2 / N / mean ** 2
    b = 1 / 2 / (N - 1)
    return a + b


def get_lnSD(sd, n):
    return np.log(sd) + (1 / (2 * (n - 1)))


def get_var_lnSD(n):
    return 1 / (2 * (n - 1))


def get_var_lnMean(mean, sd, n):
    return sd ** 2 / mean ** 2 / n
