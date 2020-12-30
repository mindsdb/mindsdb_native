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
    return softmax(torch.Tensor(x)/t, dim=axis).numpy()


def _df_from_x(x, columns):
    x = pd.DataFrame(x)
    x.columns = columns
    return x


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


def filter_cols(columns, target, ignore):
    cols = deepcopy(columns)
    for col in [target] + ignore:
        if col in cols:
            cols.remove(col)
    return cols


def get_conf_range(X, icp, target, typing_info, lmd, std_tol=1):
    """ Returns confidence and confidence ranges for predictions"""
    # numerical
    if typing_info['data_type'] == DATA_TYPES.NUMERIC or (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                                                          DATA_TYPES.NUMERIC in typing_info['data_type_dist'].keys()):
        # ICP gets all possible bounds (shape: (B, 2, 99))
        all_ranges = icp.predict(X.values)

        # iterate over confidence levels until spread >= a multiplier of the dataset stddev
        significances = [*range(0, 10), *range(10, 30, 5), *range(40, 100, 10)]
        for significance in significances:
            ranges = all_ranges[:, :, significance]
            spread = np.mean(ranges[:, 1] - ranges[:, 0])
            tolerance = lmd['stats_v2']['train_std_dev'][target] * std_tol

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
    return 0.005, np.zeros_like((X.size, 2))


class BoostedSignErrorErrFunc(RegressionErrFunc):
    """Calculates signed error nonconformity for regression problems. Applies linear interpolation
    for nonconformity scores when we have less than 100 samples in the validation dataset.
    """

    def __init__(self):
        super(BoostedSignErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return prediction - y

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        if nc.size < 100:
            x = np.arange(nc.shape[0])
            interp = interp1d(x, nc)
            nc = interp(np.linspace(0, nc.size-1, 100))
        upper = int(np.floor((significance / 2) * (nc.size + 1)))
        lower = int(np.floor((1 - significance / 2) * (nc.size + 1)))
        upper = min(max(upper, 0), nc.size - 1)
        lower = max(min(lower, nc.size - 1), 0)
        return np.vstack([-nc[lower], nc[upper]])


class ConformalRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalRegressorAdapter, self).__init__(model, fit_params)
        self.target = fit_params['target']
        self.columns = fit_params['all_columns']
        self.ignore_columns = fit_params['columns_to_ignore']
        self.ar = fit_params['use_previous_target']
        if self.ar:
            self.columns.append(f'__mdb_ts_previous_{self.target}')

    def fit(self, x, y):
        """
        :param x: numpy.array, shape (n_train, n_features)
        :param y: numpy.array, shape (n_train)
        We omit implementing this method as the Conformal Estimator is called once
        the MindsDB mixer has already been trained. However, it has to be called to
        setup some things in the nonconformist backend.
        """
        pass

    def predict(self, x):
        """
        :param x: numpy.array, shape (n_train, n_features). Raw data for predicting outputs.
        n_features = (|all_cols| - |ignored| - |target|)

        :return: output compatible with nonconformity function. For default
        ones, this should a numpy.array of shape (n_test) with predicted values
        """
        cols = filter_cols(self.columns, self.target, self.ignore_columns)
        x = _df_from_x(x, cols)
        predictions = self.model.predict(when_data=x)
        ys = np.array(predictions[self.target]['predictions'])
        return ys


class ConformalClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalClassifierAdapter, self).__init__(model, fit_params)
        self.target = fit_params['target']
        self.columns = fit_params['all_columns']
        self.ignore_columns = fit_params['columns_to_ignore']
        self.ar = fit_params['use_previous_target']
        if self.ar:
            self.columns.append(f'__mdb_ts_previous_{self.target}')

    def fit(self, x, y):
        """
        :param x: numpy.array, shape (n_train, n_features)
        :param y: numpy.array, shape (n_train)
        We omit implementing this method as the Conformal Estimator is called once
        the MindsDB mixer has already been trained. However, it has to be called to
        setup some things in the nonconformist backend.
        """
        pass

    def predict(self, x):
        """
        :param x: numpy.array, shape (n_train, n_features). Raw data for predicting outputs.
        n_features = (|all_cols| - |ignored| - |target|)

        :return: output compatible with nonconformity function. For default
        ones, this should a numpy.array of shape (n_test, n_classes) with
        class probability estimates
        """
        self.model.config['include_extra_data'] = True
        cols = filter_cols(self.columns, self.target, self.ignore_columns)
        x = _df_from_x(x, cols)
        predictions = self.model.predict(when_data=x)  # ToDo: return complete class distribution and labels from lightwood

        ys = np.array(predictions[self.target]['predictions'])
        ys = self.fit_params['one_hot_enc'].transform(ys.reshape(-1, 1))

        try:
            raw = np.array(predictions[self.target]['encoded_predictions'])
            raw_s = np.max(t_softmax(raw, t=0.5), axis=1)
            return ys * raw_s.reshape(-1, 1)
        except KeyError:
            # Not all mixers return class probabilities yet
            return ys


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
