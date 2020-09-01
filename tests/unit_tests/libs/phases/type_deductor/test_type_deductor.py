import json
from itertools import cycle
import random
from unittest import mock
from uuid import uuid4
import pytest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from mindsdb_native.libs.constants.mindsdb import DATA_TYPES, DATA_SUBTYPES
from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.helpers.stats_helpers import sample_data
from mindsdb_native.libs.phases.type_deductor.type_deductor import TypeDeductor
from unit_tests.utils import (
    test_column_types,
    generate_short_sentences,
    generate_rich_sentences,
    VOCAB
)


class TestTypeDeductor:
    @pytest.fixture()
    def lmd(self, transaction):
        lmd = transaction.lmd
        lmd['stats_v2'] = {}
        lmd['data_types'] = {}
        lmd['data_subtypes'] = {}
        lmd['data_preparation'] = {}
        lmd['force_categorical_encoding'] = []
        lmd['columns_to_ignore'] = []
        lmd['predict_columns'] = []

        lmd['sample_settings'] = dict(
            sample_for_analysis=False,
            sample_for_training=False,
            sample_margin_of_error=0.005,
            sample_confidence_level=1 - 0.005,
            sample_percentage=None,
            sample_function='sample_data'
        )

        return lmd

    def test_type_deduction(self, transaction, lmd):
        """Tests that basic cases of type deduction work correctly"""
        hmd = transaction.hmd
        lmd['handle_identifiers'] = True
        type_deductor = TypeDeductor(session=transaction.session,
                                     transaction=transaction)

        n_points = 100

        # Apparently for n_category_values = 10 it doesnt work
        n_category_values = 4
        categories_cycle = cycle(range(n_category_values))
        n_multilabel_category_values = 25
        multiple_categories_str_cycle = cycle(random.choices(VOCAB[0:20], k=n_multilabel_category_values))

        input_dataframe = pd.DataFrame({
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
        }, index=list(range(n_points)))

        input_data = TransactionData()
        input_data.data_frame = input_dataframe
        type_deductor.run(input_data)

        stats_v2 = lmd['stats_v2']

        for col_name in input_dataframe.columns:
            expected_type = test_column_types[col_name][0]
            expected_subtype = test_column_types[col_name][1]
            assert stats_v2[col_name]['typing']['data_type'] == expected_type
            assert stats_v2[col_name]['typing']['data_subtype'] == expected_subtype
            assert stats_v2[col_name]['typing']['data_type_dist'][expected_type] == 100
            assert stats_v2[col_name]['typing']['data_subtype_dist'][expected_subtype] == 100

        assert DATA_SUBTYPES.INT in stats_v2['categorical_int']['additional_info']['other_potential_subtypes']
        assert hmd == {}

        assert isinstance(json.dumps(transaction.lmd), str)

    def test_deduce_foreign_key(self, transaction, lmd):
        """Tests that basic cases of type deduction work correctly"""
        hmd = transaction.hmd

        lmd['handle_identifiers'] = True

        type_deductor = TypeDeductor(session=transaction.session,
                                     transaction=transaction)
        n_points = 100

        input_dataframe = pd.DataFrame({
            'numeric_id': list(range(n_points)),
            'uuid': [str(uuid4()) for i in range(n_points)]
        }, index=list(range(n_points)))

        input_data = TransactionData()
        input_data.data_frame = input_dataframe
        type_deductor.run(input_data)

        stats_v2 = lmd['stats_v2']

        assert stats_v2['numeric_id']['typing']['data_subtype'] == DATA_SUBTYPES.IDENTIFIER
        assert stats_v2['uuid']['typing']['data_subtype'] == DATA_SUBTYPES.IDENTIFIER

        assert 'numeric_id' in lmd['columns_to_ignore']
        assert 'uuid' in lmd['columns_to_ignore']

    def test_empty_values(self, transaction, lmd):
        lmd['handle_identifiers'] = True
        type_deductor = TypeDeductor(session=transaction.session,
                                    transaction=transaction)

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_float': np.linspace(0, n_points, n_points),
        }, index=list(range(n_points)))
        input_dataframe['numeric_float'].iloc[::2] = None
        input_data = TransactionData()
        input_data.data_frame = input_dataframe
        type_deductor.run(input_data)

        stats_v2 = lmd['stats_v2']
        assert stats_v2['numeric_float']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 50
        assert stats_v2['numeric_float']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 50

    def test_type_mix(self, transaction, lmd):
        lmd['handle_identifiers'] = True
        type_deductor = TypeDeductor(session=transaction.session,
                                     transaction=transaction)

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_float': np.linspace(0, n_points, n_points),
        }, index=list(range(n_points)))
        input_dataframe['numeric_float'].iloc[:2] = 'random string'
        input_data = TransactionData()
        input_data.data_frame = input_dataframe
        type_deductor.run(input_data)

        stats_v2 = lmd['stats_v2']
        assert stats_v2['numeric_float']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 98
        assert stats_v2['numeric_float']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 98

    def test_sample(self, transaction, lmd):
        lmd['sample_settings']['sample_for_analysis'] = True
        lmd['handle_identifiers'] = True
        transaction.hmd['sample_function'] = mock.MagicMock(wraps=sample_data)

        type_deductor = TypeDeductor(session=transaction.session,
                                     transaction=transaction)

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
        }, index=list(range(n_points)))

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        type_deductor.run(input_data)

        assert transaction.hmd['sample_function'].called

        stats_v2 = lmd['stats_v2']
        assert stats_v2['numeric_int']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int']['typing']['data_subtype'] == DATA_SUBTYPES.INT
        assert stats_v2['numeric_int']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] <= n_points
        assert stats_v2['numeric_int']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] <= n_points

        lmd['sample_settings']['sample_for_analysis'] = False
        transaction.hmd['sample_function'] = mock.MagicMock(wraps=sample_data)

        type_deductor.run(input_data)
        assert not transaction.hmd['sample_function'].called

    def test_small_dataset_no_sampling(self, transaction, lmd):
        lmd['sample_settings']['sample_for_analysis'] = True
        lmd['sample_settings']['sample_margin_of_error'] = 0.95
        lmd['sample_settings']['sample_confidence_level'] = 0.05
        lmd['handle_identifiers'] = True
        transaction.hmd['sample_function'] = mock.MagicMock(wraps=sample_data)

        type_deductor = TypeDeductor(session=transaction.session,
                                     transaction=transaction)

        n_points = 50
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
        }, index=list(range(n_points)))

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        type_deductor.run(input_data)

        assert transaction.hmd['sample_function'].called

        stats_v2 = lmd['stats_v2']
        assert stats_v2['numeric_int']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int']['typing']['data_subtype'] == DATA_SUBTYPES.INT

        # This ensures that no sampling was applied
        assert stats_v2['numeric_int']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 50
        assert stats_v2['numeric_int']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] == 50
