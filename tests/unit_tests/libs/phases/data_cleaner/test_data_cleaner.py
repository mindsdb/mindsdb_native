import unittest
import pandas as pd
import numpy as np
import random

from mindsdb_native import functional as F
from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException


class TestDataCleaner(unittest.TestCase):
    def test_ignore_columns(self):
        predictor = Predictor(name='test_ignore_columns')
        predictor.breakpoint = 'DataCleaner'

        n_points = 100

        df = pd.DataFrame({
            'do_use': [*range(n_points)],
            'ignore_this': [x % 2 for x in range(n_points)]
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

        n_points = 50

        null_data = ['NULL', 'null', 'none', 'Null']
        non_null_data = [x % 3 + 1 for x in range(n_points - len(null_data))]

        data = null_data + non_null_data
        random.shuffle(data)

        df = pd.DataFrame({
            'my_column': data,
            'to_predict': [x % 3 for x in range(n_points)]
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

        # Dont compare sets containing np.nan, because for some reason there can be two np.nan in a set
        # and tests fails even though everything works as expected
        notna_values = [x for x in predictor.transaction.input_data.data_frame['my_column'] if not pd.isna(x)]

        assert set(notna_values) == set([1, 2, 3])

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

    def test_remove_target_outliers(self):
        data = pd.DataFrame({'x': np.arange(1210),
                             'y': np.hstack([np.random.uniform(0, 1, 400),
                                             np.random.uniform(100, 1000, 800),
                                             np.random.uniform(1e4, 2e4, 10)])
                             })

        for z_score in (3, 0):
            predictor = Predictor(name=f'test_remove_target_outlier_{z_score}')
            predictor.breakpoint = 'DataCleaner'
            try:
                predictor.learn(from_data=data,
                                to_predict='y',
                                stop_training_in_x_seconds=1,
                                advanced_args={'remove_target_outliers': z_score},  # 0 -> disabled
                                use_gpu=False)
            except BreakpointException:
                pass
            else:
                raise AssertionError
            if z_score:
                assert predictor.transaction.input_data.data_frame['y'].max() <= 1000
            else:
                assert predictor.transaction.input_data.data_frame['y'].max() > 1000
