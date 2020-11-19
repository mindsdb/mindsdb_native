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
        assert len(predictions['hours-per-week']) == len(df_test)
        for pred in predictions['hours-per-week']:
            assert isinstance(pred,int)
            assert pred > 0
