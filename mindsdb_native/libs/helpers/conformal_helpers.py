from nonconformist.base import RegressorAdapter
from nonconformist.base import ClassifierAdapter
from mindsdb_native.libs.constants.mindsdb import *

import pandas as pd
import numpy as np
import copy


def _df_from_x(x, columns):
    x = pd.DataFrame(x)
    x.columns = columns
    return x


def clean_df(df, stats, output_columns):
    """
    :param stats: dict with information about every column
    :param output_columns: to be predicted
    """
    for key, value in stats.items():
        if key in df.columns and key in output_columns:
            df.pop(key)
    return df


class ConformalRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalRegressorAdapter, self).__init__(model, fit_params)
        self.target = fit_params['target']
        self.columns = fit_params['all_columns']
        self.ignore_columns = fit_params['columns_to_ignore']

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
        cols = copy.deepcopy(self.columns)
        for col in [self.target] + self.ignore_columns:
            cols.remove(col)
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
        cols = copy.deepcopy(self.columns)
        for col in [self.target] + self.ignore_columns:
            cols.remove(col)
        x = _df_from_x(x, cols)
        predictions = self.model.predict(when_data=x)
        ys = np.array(predictions[self.target]['predictions'])
        ys = self.fit_params['one_hot_enc'].transform(ys.reshape(-1, 1))  # ideally, complete class distribution here
        return ys
