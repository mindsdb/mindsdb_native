import unittest
import pandas as pd
from mindsdb_native import Predictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_iris


class TestConformal(unittest.TestCase):
    def test_regressor(self):
        """
        Sanity check. MindsDB point predictions should be within range
        of predicted bounds by the inductive conformal predictor.
        """
        def _df_from_x(x, columns=None):
            x = pd.DataFrame(x)
            if columns is None:
                x.columns = 'c' + pd.Series([i for i in range(len(x.columns))]).astype(str)
            else:
                x.columns = columns
            return x

        def _df_from_xy(x, y, target):
            x = _df_from_x(x)
            x[target] = pd.DataFrame(y)
            return x

        X, y = load_boston(return_X_y=True)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=5)
        target = 'medv'

        x_tr = _df_from_xy(X_train, Y_train, target)
        p = Predictor('ConformalTest')
        p.learn(from_data=x_tr, to_predict=target)

        x_te = _df_from_xy(X_test, Y_test, target)
        r = p.predict(when_data=x_te)
        r = [x.explanation[target] for x in r]

        for x in r:
            assert x['confidence_interval'][0] <= x['predicted_value'] <= x['confidence_interval'][1]
