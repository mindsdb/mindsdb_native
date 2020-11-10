import unittest
import json
import random
from unittest import mock
import tempfile

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, f1_score, accuracy_score

from mindsdb_native.libs.controllers.predictor import Predictor
from mindsdb_native import F

from mindsdb_native.libs.data_sources.file_ds import FileDS
from mindsdb_native.libs.constants.mindsdb import DATA_TYPES, DATA_SUBTYPES

from unit_tests.utils import (
    generate_value_cols,
    generate_timeseries_labels,
    generate_timeseries,
    columns_to_file,
)

from mindsdb_native.libs.helpers.stats_helpers import sample_data


class TestPredictorTimeseries(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_timeseries(self):
        ts_hours = 12
        data_len = 120
        train_file_name = os.path.join(self.tmp_dir, 'train_data.csv')
        test_file_name = os.path.join(self.tmp_dir, 'test_data.csv')

        features = generate_value_cols(['date', 'int'], data_len, ts_hours * 3600)
        labels = [generate_timeseries_labels(features)]

        feature_headers = list(map(lambda col: col[0], features))
        label_headers = list(map(lambda col: col[0], labels))

        # Create the training dataset and save it to a file
        columns_train = list(map(lambda col: col[1:int(len(col) * 3 / 4)], features))
        columns_train.extend(list(map(lambda col: col[1:int(len(col) * 3 / 4)], labels)))
        columns_to_file(
            columns_train,
            train_file_name,
            headers=[*feature_headers, *label_headers]
        )
        # Create the testing dataset and save it to a file
        columns_test = list(map(lambda col: col[int(len(col) * 3 / 4):], features))
        columns_to_file(
            columns_test,
            test_file_name,
            headers=feature_headers
        )

        mdb = Predictor(name='test_timeseries')

        mdb.learn(
            from_data=train_file_name,
            to_predict=label_headers,
            timeseries_settings={
                'order_by': [feature_headers[0]],
                'window': 3
            },
            stop_training_in_x_seconds=10,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        results = mdb.predict(when_data=test_file_name, use_gpu=False)

        # Results should only be given for the rows with full history
        assert len(results) == len(columns_test[-1])
        for row in results:
            expect_columns = [label_headers[0], label_headers[0] + '_confidence']
            for col in expect_columns:
                assert col in row

        for row in [x.explanation[label_headers[0]] for x in results]:
            assert row['confidence_interval'][0] <= row['predicted_value'] <= row['confidence_interval'][1]

        models = F.get_models()
        model_data = F.get_model_data(models[0]['name'])
        assert model_data

    def test_timeseries_stepahead(self):
        ts_hours = 12
        data_len = 120
        train_file_name = os.path.join(self.tmp_dir, 'train_data.csv')
        test_file_name = os.path.join(self.tmp_dir, 'test_data.csv')

        features = generate_value_cols(['date', 'int'], data_len, ts_hours * 3600)
        labels = [generate_timeseries_labels(features)]

        feature_headers = list(map(lambda col: col[0], features))
        label_headers = list(map(lambda col: col[0], labels))

        # Create the training dataset and save it to a file
        columns_train = list(map(lambda col: col[1:int(len(col) * 3 / 4)], features))
        columns_train.extend(list(map(lambda col: col[1:int(len(col) * 3 / 4)], labels)))
        columns_to_file(
            columns_train,
            train_file_name,
            headers=[*feature_headers, *label_headers]
        )
        # Create the testing dataset and save it to a file
        columns_test = list(map(lambda col: col[int(len(col) * 3 / 4):], features))
        columns_to_file(
            columns_test,
            test_file_name,
            headers=feature_headers
        )

        mdb = Predictor(name='test_timeseries')

        mdb.learn(
            from_data=train_file_name,
            to_predict=label_headers,
            timeseries_settings={
                'order_by': [feature_headers[0]]
                ,'window': 3
                ,'nr_predictions': 6
            },
            stop_training_in_x_seconds=10,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        results = mdb.predict(when_data=test_file_name, use_gpu=False)

        # Results should only be given for the rows with full history
        assert len(results) == len(columns_test[-1])

        for row in results:
            assert label_headers[0] + '_confidence' in row
            assert label_headers[0] in row
            assert isinstance(row[label_headers[0]], list)
            assert len(row[label_headers[0]]) == 6

    def test_keep_id_orderby(self):
        data_len = 100
        train_file_name = os.path.join(self.tmp_dir, 'train_data.csv')
        test_file_name = os.path.join(self.tmp_dir, 'test_data.csv')
        col_name = 'id'

        features = [generate_timeseries(data_len, period=1)]
        features[0].insert(0, col_name)
        labels = [[str(random.randint(0, 1)) for _ in range(len(features[0][1:]))]]
        labels[0].insert(0, 'Y')

        feature_headers = list(map(lambda col: col[0], features))
        label_headers = list(map(lambda col: col[0], labels))

        # Create the training dataset and save it to a file
        columns_train = list(map(lambda col: col[1:int(len(col) * 3 / 4)], features))
        columns_train.extend(list(map(lambda col: col[1:int(len(col) * 3 / 4)], labels)))
        columns_to_file(
            columns_train,
            train_file_name,
            headers=[*feature_headers, *label_headers]
        )
        # Create the testing dataset and save it to a file
        columns_test = list(map(lambda col: col[int(len(col) * 3 / 4):], features))
        columns_to_file(
            columns_test,
            test_file_name,
            headers=feature_headers
        )

        mdb = Predictor(name='test_timeseries')

        mdb.learn(
            from_data=train_file_name,
            to_predict=label_headers,
            timeseries_settings={
                'order_by': [feature_headers[0]]
                ,'window': 2
            },
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        admittable = ['Auto-incrementing identifier']
        assert col_name not in mdb.transaction.lmd['columns_to_ignore']
        assert mdb.transaction.lmd['stats_v2'][col_name]['identifier'] in admittable

    def test_keep_order(self):
        ts_hours = 12
        data_len = 120
        train_file_name = os.path.join(self.tmp_dir, 'train_data.csv')
        test_file_name = os.path.join(self.tmp_dir, 'test_data.csv')

        features = generate_value_cols(['date', 'int'], data_len, ts_hours * 3600)
        labels = [generate_timeseries_labels(features)]

        feature_headers = list(map(lambda col: col[0], features))
        label_headers = list(map(lambda col: col[0], labels))

        features.append([x for x in range(data_len)])
        features.append([x % 3 for x in range(data_len)])
        feature_headers.append('order_ai_id')
        feature_headers.append('3_valued_group_by')

        # Create the training dataset and save it to a file
        columns_train = list(
            map(lambda col: col[1:int(len(col) * 3 / 4)], features))
        columns_train.extend(
            list(map(lambda col: col[1:int(len(col) * 3 / 4)], labels)))
        columns_to_file(columns_train, train_file_name, headers=[*feature_headers,
                                                                 *label_headers])
        # Create the testing dataset and save it to a file
        columns_test = list(
            map(lambda col: col[int(len(col) * 3 / 4):], features))
        columns_to_file(columns_test, test_file_name, headers=feature_headers)

        mdb = Predictor(name='test_timeseries')

        mdb.learn(
            from_data=train_file_name,
            to_predict=label_headers,
            timeseries_settings={
                'order_by': ['order_ai_id']
                ,'window': 3
                ,'nr_predictions': 1
                ,'group_by': ['3_valued_group_by']
            },
            stop_training_in_x_seconds=10,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        results = mdb.predict(when_data=test_file_name, use_gpu=False)

        for i, row in enumerate(results):
            # Need to somehow test the internal ordering here (??)
            assert str(row['order_ai_id']) == str(columns_test[2][i])
            assert str(row['3_valued_group_by']) == str(columns_test[3][i])
