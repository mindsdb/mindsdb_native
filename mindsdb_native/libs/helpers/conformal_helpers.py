from nonconformist.base import RegressorAdapter
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import BaseScorer, RegressionErrFunc
from mindsdb_native.libs.constants.mindsdb import *

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.interpolate import interp1d
from torch.nn.functional import softmax


def t_softmax(x, t=1.0, axis=1):
    """Softmax with temperature scaling"""
    return softmax(torch.Tensor(x)/t, dim=axis).numpy()


def clean_df(df, stats, output_columns, ignored_columns):
    """
    :param stats: dict with information about every column
    :param output_columns: to be predicted
    """
    for key, value in stats.items():
        if key in df.columns and key in output_columns:
            df.pop(key)
    for col in ignored_columns:
        if col in df.columns:
            df.pop(col)
    return df


def get_conf_range(X, icp, target, typing_info, lmd, std_tol=1):
    """ Returns confidence and confidence ranges for predictions"""
    # numerical
    if typing_info['data_type'] == DATA_TYPES.NUMERIC or (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                                                          DATA_TYPES.NUMERIC in typing_info['data_type_dist'].keys()):
        # ICP gets all possible bounds (shape: (B, 2, 99))
        all_ranges = icp.predict(X.values)

        # iterate over confidence levels until spread >= a multiplier of the dataset stddev
        for significance in range(99):
            ranges = all_ranges[:, :, significance]
            spread = np.mean(ranges[:, 1] - ranges[:, 0])
            tolerance = lmd['stats_v2'][target]['train_std_dev'] * std_tol

            if spread <= tolerance:
                confidence = (99-significance)/100
                return confidence, ranges

    # categorical
    elif (typing_info['data_type'] == DATA_TYPES.CATEGORICAL or                         # categorical
            (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and                      # time-series w/ cat target
             DATA_TYPES.CATEGORICAL in typing_info['data_type_dist'].keys())) and \
            lmd['stats_v2'][target]['typing']['data_subtype'] != DATA_SUBTYPES.TAGS:    # no tag support yet

        pvals = icp.predict(X.values)
        confs = np.subtract(1, pvals.min(axis=1))
        conf = confs.mean()
        return conf, pvals

    # default
    return 0.005, np.zeros_like((X.size, 2)).reshape(1, -1)


class BoostedAbsErrorErrFunc(RegressionErrFunc):
    """Calculates absolute error nonconformity for regression problems. Applies linear interpolation
    for nonconformity scores when we have less than 100 samples in the validation dataset.
    """
    def __init__(self):
        super(BoostedAbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        if 1 < nc.size < 100:
            x = np.arange(nc.shape[0])
            interp = interp1d(x, nc)
            nc = interp(np.linspace(0, nc.size-1, 100))
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class ConformalRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalRegressorAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None

    def fit(self, x=None, y=None):
        """
        We omit implementing this method as the Conformal Estimator is called once
        the MindsDB mixer has already been trained. However, it has to be called to
        setup some things in the nonconformist backend.
        """
        pass

    def predict(self, x=None):
        """
        Same as in .fit()
        :return: np.array (n_test, n_classes) as input to the nonconformity function, has class probability estimates
        """
        return self.prediction_cache


class ConformalClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalClassifierAdapter, self).__init__(model, fit_params)
        self.prediction_cache = None

    def fit(self, x=None, y=None):
        """
        We omit implementing this method as the Conformal Estimator is called once
        the MindsDB mixer has already been trained. However, it has to be called to
        setup some things in the nonconformist backend.
        """
        pass

    def predict(self, x=None):
        """
        Same as in .fit()
        :return: np.array (n_test, n_classes) as input to the nonconformity function, has class probability estimates
        """
        return t_softmax(self.prediction_cache, t=0.5)


class SelfawareNormalizer(BaseScorer):
    def __init__(self, fit_params=None):
        super(SelfawareNormalizer, self).__init__()
        self.prediction_cache = None
        self.output_column = fit_params['output_column']

    def fit(self, x, y):
        """No fitting is needed, as we instantiate this object
        once the self-aware NN is already trained in Lightwood."""
        pass

    def score(self, true_input, y=None):
        sa_score = self.prediction_cache.get(f'{self.output_column}_selfaware_scores', None)

        if not sa_score:
            sa_score = np.ones(true_input.shape[0])  # default case, scaling factor is 1 for all predictions
        else:
            sa_score = np.array(sa_score)

        return sa_score
