import unittest
import requests
import mindsdb_native
import pandas as pd
import requests


class TestQuickInterface(unittest.TestCase):
    def test_quick_interface(self):
        # Prepare some data
        data_url = 'https://raw.githubusercontent.com/mindsdb/benchmarks/main/datasets/adult_income/adult.csv'
        df = pd.read_csv(data_url)
        df_train = df.iloc[:int(len(df)*0.8)]
        df_test = df.iloc[int(len(df)*0.8):]

        # Train a predictor with quick_learn
        predictor = mindsdb_native.Predictor(name='test_quick_interface')
        predictor.quick_learn(
            from_data=df_train,
            to_predict='hours-per-week',
            stop_training_in_x_seconds=20
        )

        # Reload with a different name to see if it's saved properly
        test_predictor = mindsdb_native.Predictor(name='test_quick_interface')
        # Assert `report_uuid` is present and reporting is disabled, here just to have it somewhere in integration tests
        assert test_predictor.report_uuid == 'no_report'

        # Make some predictions with quick_predict and make sure they look alright
        predictions = test_predictor.quick_predict(df_test)
        assert len(predictions._data['hours-per-week']) == len(df_test)
        for pred in predictions._data['hours-per-week']:
            assert isinstance(pred, int)
            assert pred > 0

    def test_quick_predict_output(self):
        df = pd.DataFrame({
            'x1': [x for x in range(100)],
            'x2': [x*2 for x in range(100)],
            'y': [y*3 for y in range(100)]
        })

        p1 = mindsdb_native.Predictor(name='test1')
        p1.learn(from_data=df, to_predict='y')
        pred1_1 = p1.predict(when_data={'x1': 3, 'x2': 5})
        pred1_2 = p1.quick_predict(when_data={'x1': 3, 'x2': 5})

        p2 = mindsdb_native.Predictor(name='test2')
        p2.quick_learn(from_data=df, to_predict='y')
        pred2_1 = p2.predict(when_data={'x1': 3, 'x2': 5})
        pred2_2 = p2.quick_predict(when_data={'x1': 3, 'x2': 5})

        assert set(pred1_1.keys()) == set(pred2_1.keys())
        assert isinstance(pred1_2, dict)
        assert isinstance(pred2_2, dict)
