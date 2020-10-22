import json
from itertools import cycle
import random
import unittest
from unittest import mock
from uuid import uuid4
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException
from mindsdb_native.libs.constants.mindsdb import DATA_TYPES, DATA_SUBTYPES
from mindsdb_native.libs.helpers.stats_helpers import sample_data
from unit_tests.utils import (
    test_column_types,
    generate_short_sentences,
    generate_rich_sentences,
    VOCAB
)


class TestTypeDeductor(unittest.TestCase):
    def test_type_deduction(self):
        """Tests that basic cases of type deduction work correctly"""
        predictor = Predictor(name='test_type_deduction')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100

        # Apparently for n_category_values = 10 it doesnt work
        n_category_values = 4
        categories_cycle = cycle(range(n_category_values))
        n_multilabel_category_values = 25
        multiple_categories_str_cycle = cycle(
            random.choices(VOCAB[0:20], k=n_multilabel_category_values)
        )

        df = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_float': np.linspace(0, n_points, n_points),
            'date_timestamp': [(datetime.now() - timedelta(minutes=int(i))).isoformat() for i in range(n_points)],
            'date_date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_points)],
            'categorical_str': [f'category_{next(categories_cycle)}' for i in range(n_points)],
            'categorical_int': [next(categories_cycle) for i in range(n_points)],
            'categorical_binary': [0, 1] * (n_points//2),
            'sequential_array': [f"1,2,3,4,5,{i}" for i in range(n_points)],
            'multiple_categories_array_str': [",".join([f'{next(multiple_categories_str_cycle)}' for j in range(random.randint(1,6))]) for i in range(n_points)],
            'short_text': generate_short_sentences(n_points),
            'rich_text': generate_rich_sentences(n_points)
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='categorical_int',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']

        for col_name in df.columns:
            expected_type = test_column_types[col_name][0]
            expected_subtype = test_column_types[col_name][1]
            assert stats_v2[col_name]['typing']['data_type'] == expected_type
            assert stats_v2[col_name]['typing']['data_subtype'] == expected_subtype
            assert stats_v2[col_name]['typing']['data_type_dist'][expected_type] == 100
            assert stats_v2[col_name]['typing']['data_subtype_dist'][expected_subtype] == 100

        for col_name in stats_v2['columns']:
            if col_name in predictor.transaction.lmd['columns_to_ignore']:
                continue
            assert stats_v2[col_name]['identifier'] is None

        assert DATA_SUBTYPES.INT in stats_v2['categorical_int']['additional_info']['other_potential_subtypes']

        try:
            json.dumps(predictor.transaction.lmd)
        except Exception:
            raise AssertionError

        assert set(predictor.transaction.lmd['stats_v2']['columns']) == set(df.columns)

    def test_deduce_foreign_key(self):
        """Tests that basic cases of type deduction work correctly"""
        predictor = Predictor(name='test_deduce_foreign_key')
        predictor.breakpoint = 'DataAnalyzer'
  
        n_points = 100

        df = pd.DataFrame({
            'numeric_id': list(range(n_points)),
            'uuid': [str(uuid4()) for i in range(n_points)],
            'to_predict': [i % 5 for i in range(n_points)]
        })

        try:
            predictor.learn(from_data=df, to_predict='to_predict')
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']

        assert isinstance(stats_v2['numeric_id']['identifier'], str)
        assert isinstance(stats_v2['uuid']['identifier'], str)

        assert 'numeric_id' in predictor.transaction.lmd['columns_to_ignore']
        assert 'uuid' in predictor.transaction.lmd['columns_to_ignore']

    def test_empty_values(self):
        predictor = Predictor(name='test_empty_values')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_float_1': np.linspace(0, n_points, n_points),
            'numeric_float_2': np.linspace(0, n_points, n_points),
            'numeric_float_3': np.linspace(0, n_points, n_points),
        })
        df['numeric_float_1'].iloc[::2] = None

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_float_3',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_float_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float_1']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 50
        assert stats_v2['numeric_float_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 50

    def test_type_mix(self):
        predictor = Predictor(name='test_type_mix')
        predictor.breakpoint = 'TypeDeductor'
       
        n_points = 100
        df = pd.DataFrame({
            'numeric_float_1': np.linspace(0, n_points, n_points),
            'numeric_float_2': np.linspace(0, n_points, n_points),
            'numeric_float_3': np.linspace(0, n_points, n_points),
        })
        df['numeric_float_1'].iloc[:2] = 'random string'

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_float_3',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError


        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_float_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float_1']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 98
        assert stats_v2['numeric_float_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 98

    def test_sample(self):
        sample_settings = {
            'sample_for_analysis': True,
            'sample_function': sample_data
        }
        sample_settings['sample_function'] = mock.MagicMock(wraps=sample_data)
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_sample_1')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_int_1': [x % 10 for x in list(range(n_points))],
            'numeric_int_2': [x % 10 for x in list(range(n_points))]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert sample_settings['sample_function'].called

        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_int_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int_1']['typing']['data_subtype'] == DATA_SUBTYPES.INT
        assert stats_v2['numeric_int_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] <= n_points
        assert stats_v2['numeric_int_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] <= n_points

        sample_settings = {
            'sample_for_analysis': False,
            'sample_function': sample_data
        }
        sample_settings['sample_function'] = mock.MagicMock(wraps=sample_data)
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_sample_2')
        predictor.breakpoint = 'TypeDeductor'

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert not sample_settings['sample_function'].called

    def test_small_dataset_no_sampling(self):
        sample_settings = {
            'sample_for_analysis': False,
            'sample_function': mock.MagicMock(wraps=sample_data)
        }
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_small_dataset_no_sampling')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 50
        df = pd.DataFrame({
            'numeric_int_1': [*range(n_points)],
            'numeric_int_2': [*range(n_points)],
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert not sample_settings['sample_function'].called

        stats_v2 = predictor.transaction.lmd['stats_v2']

        assert stats_v2['numeric_int_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int_1']['typing']['data_subtype'] == DATA_SUBTYPES.INT

        # This ensures that no sampling was applied
        assert stats_v2['numeric_int_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == n_points
        assert stats_v2['numeric_int_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] == n_points
