from nonconformist.base import RegressorAdapter
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import BaseScorer
from mindsdb_native.libs.constants.mindsdb import *

import torch
import numpy as np
import pandas as pd
from copy import deepcopy
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


def get_significance_level(X, Y, icp, target, typing_info, lmd):
    # numerical
    if typing_info['data_type'] == DATA_TYPES.NUMERIC or (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                                                          DATA_TYPES.NUMERIC in typing_info['data_type_dist'].keys()):
        # ICP gets all possible bounds
        all_ranges = icp.predict(X.values)
        min_error = 1
        confidence = 0

        # iterate over possible confidence levels until
        # spread equals or surpasses some multiplier of datasets standard deviation
        # TODO use only these levels
        # l = list(range(0, 10)) + list(range(10, 30, 5)) + list(range(40, 100, 10))
        # conf is then [(100-i)/100 for i in l]

        for significance in range(all_ranges.shape[2]):
            ranges = all_ranges[:, :, significance]  # shape (B, 2)
            within = ((Y >= ranges[:, 0]) & (Y <= ranges[:, 1]))
            acc = sum(within)/len(within)
            error = abs(acc - significance/100)
            if error <= min_error:
                min_error = error
                confidence = 1-significance/100
            else:
                return confidence
        else:
            return 0

    # categorical
    elif (typing_info['data_type'] == DATA_TYPES.CATEGORICAL or                         # categorical
            (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and                      # time-series w/ cat target
             DATA_TYPES.CATEGORICAL in typing_info['data_type_dist'].keys())) and \
            lmd['stats_v2'][target]['typing']['data_subtype'] != DATA_SUBTYPES.TAGS:    # no tag support yet

        # max permitted error rate
        significances = list(range(20)) + list(range(20, 100, 10))
        all_ranges = np.array(
            [icp.predict(df.values, significance=s / 100)
             for s in significances])
        lmd['all_conformal_ranges'][target] = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)

        for sample_idx in range(lmd['all_conformal_ranges'][target].shape[0]):
            sample = lmd['all_conformal_ranges'][target][sample_idx, :, :]
            for idx in range(sample.shape[1]):
                conf = (99 - significances[idx]) / 100
                if np.sum(sample[:, idx]) == 1:
                    output_data[f'{target}_confidence'][sample_idx] = conf
                    break
            else:
                output_data[f'{target}_confidence'][sample_idx] = 0.005


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
