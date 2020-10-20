import pandas as pd
import numpy as np
import unittest

from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException
from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.phases.data_cleaner.data_cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    def test_ignore_columns(self):
        predictor = Predictor(name='test_ignore_columns')

        df = pd.DataFrame({
            'do_use': [1, 2, 3],
            'ignore_this': [0, 1, 100]
        })

        predictor.breakpoint = 'DataCleaner'

        try:
            predictor.learn(
                from_data=df,
                to_predict='do_use',
                ignore_columns=['ignore_this']
            )
        except BreakpointException:
            pass
        else:
            assert False

        assert 'do_use' in predictor.transaction.input_data.data_frame.columns
        assert 'ignore_this' not in predictor.transaction.input_data.data_frame.columns

    def test_user_provided_null_values(self):
        predictor = Predictor(name='test_null_values')

        df = pd.DataFrame({
            'my_column': ['a', 'b', 'NULL', 'c', 'null', 'none', 'Null']
            'to_predict': [1, 2, 3, 1, 2, 3, 1]
        })

        predictor.learn(
            from_data=df,
            to_predict='to_predict',
            advanced_args={
                'null_values': {
                    'my_column': ['NULL', 'null', 'none', 'Null']
                }
            }
        )

        assert predictor.transaction.input_data.data_frame['my_column'].iloc[0] == 'a'
        assert predictor.transaction.input_data.data_frame['my_column'].iloc[1] == 'b'
        assert pd.isna(predictor.transaction.input_data.data_frame['my_column'].iloc[2])
        assert predictor.transaction.input_data.data_frame['my_column'].iloc[3] == 'c'
        assert pd.isna(predictor.transaction.input_data.data_frame['my_column'].iloc[4])
        assert pd.isna(predictor.transaction.input_data.data_frame['my_column'].iloc[5])
        assert pd.isna(predictor.transaction.input_data.data_frame['my_column'].iloc[6])
