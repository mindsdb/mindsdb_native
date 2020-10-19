import json

import pytest
from unittest import mock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.helpers.stats_helpers import sample_data
from mindsdb_native.libs.phases.data_analyzer.data_analyzer import DataAnalyzer
from unit_tests.utils import (
    test_column_types,
    generate_short_sentences,
    generate_rich_sentences
)


class TestDataAnalyzer:
    @pytest.fixture(scope='function')
    def lmd(self, transaction):
        lmd = transaction.lmd
        lmd['stats_v2'] = {}
        lmd['empty_columns'] = []
        lmd['data_types'] = {}
        lmd['data_subtypes'] = {}
        lmd['data_preparation'] = {}
        lmd['force_categorical_encoding'] = []
        lmd['columns_to_ignore'] = []

        lmd['sample_settings'] = dict(
            sample_for_analysis=False,
            sample_for_training=False,
            sample_margin_of_error=0.005,
            sample_confidence_level=1 - 0.005,
            sample_percentage=None,
            sample_function='sample_data'
        )

        return lmd

    def get_stats_v2(self, col_names):
        result = {}
        for col_name, col_type_pair in test_column_types.items():
            if col_name in col_names:
                result[col_name] = {
                    'typing': {
                        'data_type': col_type_pair[0],
                        'data_subtype': col_type_pair[1],
                    }
                }

        for k, v in result.items():
            result[k]['typing']['data_type_dist'] = {v['typing']['data_type']: 100}
            result[k]['typing']['data_subtype_dist'] = {v['typing']['data_subtype']: 100}
        return result

    def test_data_analysis(self, transaction, lmd):
        """Tests that data analyzer doesn't crash on common types"""
        data_analyzer = DataAnalyzer(session=transaction.session,
                                     transaction=transaction)

        n_points = 100
        n_category_values = 4
        input_dataframe = pd.DataFrame({
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

        }, index=list(range(n_points)))

        stats_v2 = self.get_stats_v2(input_dataframe.columns)

        lmd['stats_v2'] = stats_v2
        hmd = transaction.hmd

        input_data = TransactionData()
        input_data.data_frame = input_dataframe
        data_analyzer.run(input_data)

        stats_v2 = lmd['stats_v2']

        for col_name in input_dataframe.columns:
            assert stats_v2[col_name]['empty']['empty_percentage'] == 0
            assert not stats_v2[col_name]['empty']['is_empty']
            assert stats_v2[col_name]['histogram']
            assert 'percentage_buckets' in stats_v2[col_name]
            assert stats_v2[col_name]['bias']['entropy']

        assert stats_v2['categorical_str']['unique']['unique_values']
        assert stats_v2['categorical_str']['unique']['unique_percentage'] == 4.0

        # Assert that the histogram on text field is made using words
        assert isinstance(stats_v2['short_text']['histogram']['x'][0], str)
        assert isinstance(stats_v2['rich_text']['histogram']['x'][0], str)

        for col in ['numeric_float', 'numeric_int']:
            assert isinstance(stats_v2[col]['outliers']['outlier_values'], list)
            assert isinstance(stats_v2[col]['outliers']['outlier_buckets'], list)
            assert isinstance(stats_v2[col]['outliers']['description'], str)
            assert set(stats_v2[col]['outliers']['outlier_buckets']) <= set(stats_v2[col]['percentage_buckets'])

        assert hmd == {}

        assert isinstance(json.dumps(transaction.lmd), str)

    def test_empty_values(self, transaction, lmd):
        data_analyzer = DataAnalyzer(session=transaction.session,
                                     transaction=transaction)

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
        }, index=list(range(n_points)))

        stats_v2 = self.get_stats_v2(input_dataframe.columns)

        lmd['stats_v2'] = stats_v2

        input_dataframe['numeric_int'].iloc[::2] = None
        input_data = TransactionData()
        input_data.data_frame = input_dataframe
        data_analyzer.run(input_data)

        stats_v2 = lmd['stats_v2']

        assert stats_v2['numeric_int']['empty']['empty_percentage'] == 50

    def test_sample(self, transaction, lmd):
        lmd['sample_settings']['sample_for_analysis'] = True
        transaction.hmd['sample_function'] = mock.MagicMock(wraps=sample_data)

        data_analyzer = DataAnalyzer(session=transaction.session,
                                     transaction=transaction)

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
        }, index=list(range(n_points)))

        stats_v2 = self.get_stats_v2(input_dataframe.columns)
        lmd['stats_v2'] = stats_v2

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        data_analyzer.run(input_data)
        assert transaction.hmd['sample_function'].called

        assert sum(lmd['stats_v2']['numeric_int']['histogram']['y']) <= n_points

        lmd['sample_settings']['sample_for_analysis'] = False
        transaction.hmd['sample_function'] = mock.MagicMock(wraps=sample_data)

        data_analyzer.run(input_data)
        assert not transaction.hmd['sample_function'].called

    def test_guess_probability(self, transaction, lmd):
        data_analyzer = DataAnalyzer(
            session=transaction.session,
            transaction=transaction
        )

        input_dataframe = pd.DataFrame({
            'categorical_int': [1, 2, 1, 3, 4, 3, 2, 4, 5, 1, 2, 3],
            'categorical_int': [2, 1, 3, 4, 3, 2, 4, 5, 1, 2, 1, 2],
            'categorical_binary': ['cat', 'cat', 'cat', 'dog', 'dog', 'cat', 'cat', 'cat', 'cat', 'cat', 'cat', 'dog']
        }, index=[*range(12)])

        stats_v2 = self.get_stats_v2(input_dataframe.columns)
        lmd['stats_v2'] = stats_v2

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        data_analyzer.run(input_data)
        assert data_analyzer.transaction.lmd['stats_v2']['categorical_binary']['guess_probability'] == (9 / 12)**2 + (3 / 12)**2
