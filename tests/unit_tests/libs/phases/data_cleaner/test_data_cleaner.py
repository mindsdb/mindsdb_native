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
        predictor.breakpoint = 'DataCleaner'

        df = pd.DataFrame({
            'do_use': [1, 2, 3],
            'ignore_this': [0, 1, 100]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='do_use',
                ignore_columns=['ignore_this']
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert 'do_use' in predictor.transaction.input_data.data_frame.columns
        assert 'ignore_this' not in predictor.transaction.input_data.data_frame.columns

    def test_user_provided_null_values(self):
        predictor = Predictor(name='test_user_provided_null_values')
        predictor.breakpoint = 'DataCleaner'

        df = pd.DataFrame({
            'my_column': ['a', 'b', 'NULL', 'c', 'null', 'none', 'Null'],
            'to_predict': [1, 2, 3, 1, 2, 3, 1]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='to_predict',
                advanced_args={
                    'null_values': {
                        'my_column': ['NULL', 'null', 'none', 'Null']
                    }
                }
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError
        
        assert set(predictor.transaction.input_data.data_frame['my_column']) == set(['a', 'b', 'c', np.nan])
