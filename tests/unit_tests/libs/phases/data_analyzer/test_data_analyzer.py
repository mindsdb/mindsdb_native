import json

import unittest
from unittest import mock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from mindsdb_native.libs.controllers.functional import analyse_dataset
from mindsdb_native.libs.controllers.transaction import BreakpointException
from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.helpers.stats_helpers import sample_data
from mindsdb_native.libs.phases.data_analyzer.data_analyzer import DataAnalyzer
from unit_tests.utils import (
    test_column_types,
    generate_short_sentences,
    generate_rich_sentences
)


class TestDataAnalyzer(unittest.TestCase):
    def test_data_analysis(self):
        n_points = 100
        n_category_values = 4
        df = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_float': np.linspace(0, n_points, n_points),
            'date_timestamp': [(datetime.now() - timedelta(minutes=int(i))).isoformat() for i in range(n_points)],
            'date_date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_points)],
            'categorical_str': [f'a{x}' for x in (list(range(n_category_values)) * (n_points//n_category_values))],
            'categorical_binary': [0, 1] * (n_points//2),
            'categorical_int': [x for x in (list(range(n_category_values)) * (n_points // n_category_values))],
            'sequential_array': [f"1,2,3,4,5,{i}" for i in range(n_points)],
            'short_text': generate_short_sentences(n_points),
            'rich_text': generate_rich_sentences(n_points)
        })

        stats = analyse_dataset(df)['data_analysis_v2']

        for col_name in df.columns:
            assert stats[col_name]['empty']['empty_percentage'] == 0
            assert not stats[col_name]['empty']['is_empty']
            assert stats[col_name]['histogram']
            assert 'percentage_buckets' in stats[col_name]
            assert stats[col_name]['bias']['entropy']

        assert stats['categorical_str']['unique']['unique_values']
        assert stats['categorical_str']['unique']['unique_percentage'] == 4.0

        # Assert that the histogram on text field is made using words
        assert isinstance(stats['short_text']['histogram']['x'][0], str)
        assert isinstance(stats['rich_text']['histogram']['x'][0], str)

        for col in ['numeric_float', 'numeric_int']:
            assert isinstance(stats[col]['outliers']['outlier_values'], list)
            assert isinstance(stats[col]['outliers']['outlier_buckets'], list)
            assert isinstance(stats[col]['outliers']['description'], str)
            assert set(stats[col]['outliers']['outlier_buckets']) <= set(stats[col]['percentage_buckets'])

    def test_empty_values(self):
        df = pd.DataFrame({'numeric_int': [x % 10 for x in [*range(100)]]})
        df['numeric_int'].iloc[::2] = None

        stats = analyse_dataset(df)['data_analysis_v2']
    
        assert stats['numeric_int']['empty']['empty_percentage'] == 50

    def test_sample_true(self):
        N = 100
        df = pd.DataFrame({'numeric_int': [x % 10 for x in [*range(N)]]})
        df['numeric_int'].iloc[::2] = None

        sample_settings = {
            'sample_function': mock.MagicMock(wraps=sample_data),
            'sample_for_analysis': True
        }
        setattr(sample_settings['sample_function'], '__name__', sample_data)

        stats = analyse_dataset(df, sample_settings)['data_analysis_v2']

        assert sample_settings['sample_function'].called
        assert sum(stats['numeric_int']['histogram']['y']) <= N

    def test_sample_false(self):
        N = 100
        df = pd.DataFrame({'numeric_int': [x % 10 for x in [*range(N)]]})
        df['numeric_int'].iloc[::2] = None

        sample_settings = {
            'sample_function':mock.MagicMock(wraps=sample_data),
            'sample_for_analysis': False
        }
        setattr(sample_settings['sample_function'], '__name__', sample_data)

        stats = analyse_dataset(df, sample_settings)['data_analysis_v2']

        assert not sample_settings['sample_function'].called
        assert sum(stats['numeric_int']['histogram']['y']) <= N

    def test_guess_probability(self):
        df = pd.DataFrame({
            'categorical_int': [1, 2, 1, 3, 4, 3, 2, 4, 5, 1, 2, 3],
            'categorical_int': [2, 1, 3, 4, 3, 2, 4, 5, 1, 2, 1, 2],
            'categorical_binary': ['cat', 'cat', 'cat', 'dog', 'dog', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog']
        })

        stats = analyse_dataset(df)['data_analysis_v2']

        assert stats['categorical_binary']['guess_probability'] == (9 / 12)**2 + (3 / 12)**2
