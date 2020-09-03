import unittest
import pandas as pd
from mindsdb_native import Predictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_iris


class TestEvaluateAccuracy(unittest.TestCase):

    def _df_from_x(self, x, columns=None):
        x = pd.DataFrame(x)
        if columns is None:
            x.columns = 'c' + pd.Series([i for i in range(len(x.columns))]).astype(str)
        else:
            x.columns = columns
        return x

    def _df_from_xy(self, x, y, target):
        x = self._df_from_x(x)
        x[target] = pd.DataFrame(y)
        return x

    def test_regressor(self):
        """
        Sanity check. Passes if point predictions of MindsDB fall within the range
        of predicted values by the Inductive Conformal Predictor.
        """
        X, y = load_boston(return_X_y=True)
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=5)
        target = 'medv'

        x_tr = self._df_from_xy(X_train, Y_train, target)
        self.p = Predictor("ConformalTest")
        self.p.learn(from_data=x_tr, to_predict=target)

        x_te = self._df_from_xy(X_test, Y_test, target)
        r = self.p.predict(when_data=x_te)
        r = [x.explanation[target] for x in r]

        for x in r:
            self.assertTrue(x['confidence_interval'][0] <= x['predicted_value'] <= x['confidence_interval'][1])

    def test_home_rentals(self):
        """
        Sanity check. Passes if point predictions of MindsDB fall within the range
        of predicted values by the Inductive Conformal Predictor.
        """
        df = pd.read_csv("https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv")
        x_tr, x_te = train_test_split(df, test_size=0.1)
        target = 'rental_price'

        x_tr = self._df_from_x(x_tr, columns=df.columns)
        self.p = Predictor("ConformalTest")
        self.p.learn(from_data=x_tr, to_predict=target)

        x_te = self._df_from_x(x_te, columns=df.columns)
        r = self.p.predict(when_data=x_te)
        r = [x.explanation[target] for x in r]

        for x in r:
            self.assertTrue(x['confidence_interval'][0] <= x['predicted_value'] <= x['confidence_interval'][1])


if __name__ == '__main__':
    unittest.main()
