import pandas as pd
import numpy as np
import unittest

from mindsdb_native import functional as F
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

    def test_ignore_identifiers(self):
        df = pd.DataFrame({
            'do_use': [*range(60), *range(40)],
            'numeric_id': list(range(100)),
            'malicious_naming': list(range(99)) + [200],
            'y': list(range(100)),
        })

        predictor = Predictor(name='test_ignore_identifiers')
        predictor.breakpoint = 'DataCleaner'

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
        predictor.breakpoint = 'DataCleaner'
    
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

    def test_predictor_deduplicate_data_true(self):
        n_points = 100
        df = pd.DataFrame({
            'numeric_int': [x % 44 for x in list(range(n_points))],
            'numeric_int_2': [x % 20 for x in list(range(n_points))],
        })
        df['y'] = df['numeric_int'] % 10

        # Add duplicate row
        df = df.append(
            df.iloc[99],
            ignore_index=True
        )

        predictor = Predictor(name='test_predictor_deduplicate_data_true')
        predictor.learn(
            from_data=df,
            to_predict='y',
            stop_training_in_x_seconds=1,
            use_gpu=False
        )

        model_data = F.get_model_data('test_drop_duplicates')

        # Ensure duplicate row was not used for training, or analysis

        assert model_data['data_preparation']['total_row_count'] == n_points
        assert model_data['data_preparation']['used_row_count'] <= n_points

        assert sum([model_data['data_preparation']['train_row_count'],
                    model_data['data_preparation']['validation_row_count'],
                    model_data['data_preparation']['test_row_count']]) == n_points

        assert sum([predictor.transaction.input_data.train_df.shape[0],
                    predictor.transaction.input_data.test_df.shape[0],
                    predictor.transaction.input_data.validation_df.shape[0]]) == n_points

    def test_predictor_deduplicate_data_false(self):
        n_points = 100
        df = pd.DataFrame({
            'numeric_int': [x % 44 for x in list(range(n_points))],
            'numeric_int_2': [x % 20 for x in list(range(n_points))],
        })
        df['y'] = df['numeric_int'] % 10

        # Add duplicate row
        df = df.append(
            df.iloc[99],
            ignore_index=True
        )

        # Disable deduplication and ensure the duplicate row is used
        predictor = Predictor(name='test_predictor_deduplicate_data_false')
        predictor.learn(
            from_data=df,
            to_predict='y',
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={
                'deduplicate_data': False
            }
        )

        model_data = F.get_model_data('test_drop_duplicates')

        # Duplicate row was used for analysis and training

        assert model_data['data_preparation']['total_row_count'] == n_points + 1
        assert model_data['data_preparation']['used_row_count'] <= n_points + 1

        assert sum([model_data['data_preparation']['train_row_count'],
                    model_data['data_preparation']['validation_row_count'],
                    model_data['data_preparation']['test_row_count']]) == n_points + 1

        assert sum([predictor.transaction.input_data.train_df.shape[0],
                    predictor.transaction.input_data.test_df.shape[0],
                    predictor.transaction.input_data.validation_df.shape[0]]) == n_points + 1
