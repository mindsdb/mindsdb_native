import unittest
import pandas as pd
import numpy as np

from mindsdb_native import functional as F
from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException


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

    def test_ignore_identifiers(self):
        df = pd.DataFrame({
            'do_use': [*range(60), *range(40)],
            'numeric_id': list(range(100)),
            'malicious_naming': list(range(99)) + [200],
            'y': list(range(100)),
        })

        predictor = Predictor(name='test_ignore_identifiers')
        predictor.breakpoint = 'DataSplitter'

        try:
            predictor.learn(
                from_data=df,
                to_predict='y',
                stop_training_in_x_seconds=1,
                use_gpu=False
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert 'do_use' in predictor.transaction.input_data.train_df.columns
        # Foreign key is ignored and removed from data frames
        assert 'numeric_id' not in predictor.transaction.input_data.train_df.columns
        assert 'numeric_id' in predictor.transaction.lmd['columns_to_ignore']
        assert 'malicious_naming' not in predictor.transaction.input_data.train_df.columns
        assert 'malicious_naming' in predictor.transaction.lmd['columns_to_ignore']

    def test_force_identifier_usage(self):
        df = pd.DataFrame({
            'do_use': [*range(60), *range(40)],
            'numeric_id': list(range(100)),
            'malicious_naming': list(range(99)) + [200],
            'y': list(range(100)),
        })

        predictor = Predictor(name='test_force_identifier_usage')
        predictor.breakpoint = 'DataSplitter'
    
        try:
            predictor.learn(
                from_data=df,
                to_predict='y',
                stop_training_in_x_seconds=1,
                advanced_args={'force_column_usage': ['numeric_id']},
                use_gpu=False
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert 'do_use' in predictor.transaction.input_data.train_df.columns
        assert 'numeric_id' in predictor.transaction.input_data.train_df.columns
        assert 'numeric_id' not in predictor.transaction.lmd['columns_to_ignore']
