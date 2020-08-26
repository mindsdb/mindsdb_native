from nonconformist.base import RegressorAdapter
from nonconformist.base import ClassifierAdapter

import pandas as pd
import numpy as np


def _df_from_x(x, columns):
    x = pd.DataFrame(x)
    x.columns = columns
    return x


class ConformalRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalRegressorAdapter, self).__init__(model, fit_params)
        self.mdb_pred = model
        self.target = fit_params['target']
        self.columns = fit_params['columns']  # including target

    def fit(self, x, y):
        """
        :param x: numpy.array, shape (n_train, n_features)
        :param y: numpy.array, shape (n_train)
        We omit implementing this method as the Conformal Estimator is called once
        the MindsDB mixer has already been trained.
        """
        pass

    def predict(self, x):
        """
        :param x: numpy.array, shape (n_train, n_features). Raw data for predicting outputs.

        :return: output compatible with nonconformity function. For default
        ones, this should a numpy.array of shape (n_test) with predicted values
        """
        cols = [c for c in self.columns]
        if x.shape[-1] < len(cols):
            cols.remove(self.target)
        x = _df_from_x(x, cols)
        predictions = self.mdb_pred.predict(when_data=x)
        ys = np.array(predictions[self.target]['predictions'])
        return ys


class ConformalClassifierAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(ConformalClassifierAdapter, self).__init__(model, fit_params)
        self.mdb_pred = model
        self.target = fit_params['target']
        self.columns = fit_params['columns']  # including target

    def fit(self, x, y):
        """
        :param x: numpy.array, shape (n_train, n_features)
        :param y: numpy.array, shape (n_train)
        We omit implementing this method as the Conformal Estimator is called once
        the MindsDB mixer has already been trained.
        """
        pass

    def predict(self, x):
        """
        :param x: numpy.array, shape (n_train, n_features). Raw data for predicting outputs.

        :return: output compatible with nonconformity function. For default
        ones, this should a numpy.array of shape (n_test, n_classes) with
        class probability estimates
        """
        cols = [c for c in self.columns]
        if x.shape[-1] < len(cols):
            cols.remove(self.target)
        x = _df_from_x(x, cols)
        predictions = self.mdb_pred.predict(when_data=x)
        ys = np.array(predictions[self.target]['predictions'])
        ys = self.fit_params['one_hot_enc'].transform(ys.reshape(-1, 1))  # ideally, complete class distribution here
        return ys