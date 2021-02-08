import unittest
import tempfile
import json
import random
from unittest import mock

import os
import numpy as np
import pandas as pd

import torch
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, f1_score, accuracy_score
from sklearn.datasets import load_iris

from lightwood.config.config import CONFIG as LIGHTWOOD_CONFIG

from mindsdb_native import F
from mindsdb_native.libs.data_sources.file_ds import FileDS
from mindsdb_native.libs.controllers.predictor import Predictor
from mindsdb_native.libs.helpers.stats_helpers import sample_data
from mindsdb_native.libs.constants.mindsdb import DATA_TYPES, DATA_SUBTYPES

from unit_tests.utils import (
    generate_value_cols,
    generate_timeseries_labels,
    generate_log_labels,
    columns_to_file,
    SMALL_VOCAB
)


class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_explain_prediction(self):
        mdb = Predictor(name='test_explain_prediction')

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_x': list(range(n_points)),
            'categorical_x': [int(x % 2 == 0) for x in range(n_points)],
        }, index=list(range(n_points)))

        input_dataframe['numeric_y'] = input_dataframe.numeric_x + 2*input_dataframe.categorical_x

        mdb.learn(
            from_data=input_dataframe,
            to_predict='numeric_y',
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        # Test predicting using a data frame
        result = mdb.predict(when_data=pd.DataFrame([{"numeric_x": 10, 'categorical_x': 1}]))
        explanation_new = result[0].explanation['numeric_y']
        assert isinstance(explanation_new['predicted_value'], int)
        assert isinstance(explanation_new['confidence_interval'],list)
        assert isinstance(explanation_new['confidence_interval'][0],float)
        assert isinstance(explanation_new['important_missing_information'], list)
        assert isinstance(explanation_new['prediction_quality'], str)

        assert len(str(result[0])) > 20

    def test_data_source_setting(self):
        data_url = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/classics/german_credit_data/processed_data/test.csv'
        data_source = FileDS(data_url)
        data_source.set_subtypes({})

        data_source_mod = FileDS(data_url)
        data_source_mod.set_subtypes({
            'credit_usage': 'Int',
            'Average_Credit_Balance': 'Short Text',
            'existing_credits': 'Binary Category'
        })

        analysis = F.analyse_dataset(data_source)
        analysis_mod = F.analyse_dataset(data_source_mod)

        a1 = analysis['data_analysis_v2']
        a2 = analysis_mod['data_analysis_v2']
        assert (len(a1) == len(a2))
        assert (a1['over_draft']['typing']['data_type'] ==
                a2['over_draft']['typing']['data_type'])

        assert (a1['credit_usage']['typing']['data_type'] ==
                a2['credit_usage']['typing']['data_type'])
        assert (a1['credit_usage']['typing']['data_subtype'] !=
                a2['credit_usage']['typing']['data_subtype'])
        assert (a2['credit_usage']['typing']['data_subtype'] == DATA_SUBTYPES.INT)

        assert (a1['Average_Credit_Balance']['typing']['data_type'] !=
                a2['Average_Credit_Balance']['typing']['data_type'])
        assert (a1['Average_Credit_Balance']['typing']['data_subtype'] !=
                a2['Average_Credit_Balance']['typing']['data_subtype'])
        assert (a2['Average_Credit_Balance']['typing'][
                    'data_subtype'] == DATA_SUBTYPES.SHORT)
        assert (a2['Average_Credit_Balance']['typing'][
                    'data_type'] == DATA_TYPES.TEXT)

        assert (a1['existing_credits']['typing']['data_type'] ==
                a2['existing_credits']['typing']['data_type'])
        assert (a1['existing_credits']['typing']['data_subtype'] !=
                a2['existing_credits']['typing']['data_subtype'])
        assert (a2['existing_credits']['typing'][
                    'data_subtype'] == DATA_SUBTYPES.SINGLE)

    def test_multilabel_prediction(self):
        train_file_name = os.path.join(self.tmp_dir, 'train_data.csv')
        test_file_name = os.path.join(self.tmp_dir, 'test_data.csv')
        data_len = 60

        features = generate_value_cols(['int', 'float', 'int', 'float'], data_len)
        labels = []
        labels.append(generate_log_labels(features))
        labels.append(generate_timeseries_labels(features))

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

        mdb = Predictor(name='test_multilabel_prediction')
        mdb.learn(
            from_data=train_file_name,
            to_predict=label_headers,
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        results = mdb.predict(when_data=test_file_name)
        model_data = F.get_model_data('test_multilabel_prediction')
        assert model_data

        for i in range(len(results)):
            row = results[i]
            for label in label_headers:
                expect_columns = [label, label + '_confidence']
                for col in expect_columns:
                    assert col in row

    def test_house_pricing(self):
        self._test_house_pricing('home_rentals_cpu', use_gpu=False)
        if LIGHTWOOD_CONFIG.USE_CUDA:
            self._test_house_pricing('home_rentals_gpu', use_gpu=True)
        else:
            pass
            # with self.assertRaises(Exception):
            #     self._test_house_pricing('home_rentals_gpu_exception', use_gpu=True)

    def assert_prediction_interface(self, predictions):
        for prediction in predictions:
            assert hasattr(prediction, 'explanation')

    def _test_house_pricing(self, name, use_gpu):
        """
        Tests whole pipeline from downloading the dataset to making predictions and explanations.
        """
        predictor = Predictor(name=name)
        # Create & Learn
        predictor.learn(
            to_predict='rental_price',
            from_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
            backend='lightwood',
            stop_training_in_x_seconds=80,
            use_gpu=use_gpu
        )

        test_results = predictor.test(
            when_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
            accuracy_score_functions=r2_score,
            predict_args={'use_gpu': use_gpu}
        )
        assert test_results['rental_price_accuracy'] >= 0.8

        predictions = predictor.predict(
            when_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
            use_gpu=use_gpu
        )
        self.assert_prediction_interface(predictions)
        predictions = predictor.predict(when_data={'sqft': 300}, use_gpu=use_gpu)
        self.assert_prediction_interface(predictions)

        amd = F.get_model_data(name)
        assert isinstance(json.dumps(amd), str)

        for k in ['status', 'name', 'version', 'current_phase',
                  'updated_at', 'created_at', 'train_end_at']:
            assert isinstance(amd[k], str)

        assert isinstance(amd['predict'], (list, str))
        assert isinstance(amd['is_active'], bool)

        for k in ['validation_set_accuracy', 'accuracy_r2']:
            assert isinstance(amd[k], float)

        for k in amd['data_preparation']:
            assert isinstance(amd['data_preparation'][k], (int, float))

        model_analysis = amd['model_analysis']
        assert (len(model_analysis) > 0)
        assert isinstance(model_analysis[0], dict)
        input_importance = model_analysis[0]["overall_input_importance"]
        assert (len(input_importance) > 0)
        assert isinstance(input_importance, dict)

        for k in ['train', 'test', 'valid']:
            assert isinstance(model_analysis[0][k + '_data_accuracy'], dict)
            assert len(model_analysis[0][k + '_data_accuracy']) == 1
            assert model_analysis[0][k + '_data_accuracy']['rental_price'] > 0.4

        for column, importance in zip(input_importance["x"], input_importance["y"]):
            assert isinstance(column, str)
            assert (len(column) > 0)
            assert isinstance(importance, (float, int))
            assert (importance >= 0 and importance <= 10)

        # Check whether positive numerical domain was detected
        assert predictor.transaction.lmd['stats_v2']['rental_price']['positive_domain']

        # Check no negative predictions are emitted
        for i in (-500, -100, -10):
            neg_pred_candidate = predictor.predict(when_data={'initial_price': i}, use_gpu=use_gpu)
            assert neg_pred_candidate._data['rental_price'][0] >= 0
            assert neg_pred_candidate._data['rental_price_confidence_range'][0][0] >= 0
            assert neg_pred_candidate._data['rental_price_confidence_range'][0][1] >= 0

        # Test confidence estimation after save -> load
        F.export_predictor(name)
        try:
            F.delete_model(f'{name}-new')
        except Exception:
            pass
        F.import_model(f'{name}.zip', f'{name}-new')
        p = Predictor(name=f'{name}-new')
        predictions = p.predict(when_data={'sqft': 1000}, use_gpu=use_gpu)
        self.assert_prediction_interface(predictions)
        F.delete_model(f'{name}-new')

    def test_category_tags_input(self):
        vocab = random.sample(SMALL_VOCAB, 10)
        # tags contains up to 2 randomly selected tags
        # y contains the sum of indices of tags
        # the dataset should be nearly perfectly predicted
        n_points = 5000
        tags = []
        y = []
        for i in range(n_points):
            row_tags = []
            row_y = 0
            for k in range(2):
                if random.random() > 0.2:
                    selected_index = random.randint(0, len(vocab) - 1)
                    if vocab[selected_index] not in row_tags:
                        row_tags.append(vocab[selected_index])
                        row_y += selected_index
            tags.append(','.join(row_tags))
            y.append(row_y)

        df = pd.DataFrame({'tags': tags, 'y': y})

        df_train = df.iloc[:round(n_points * 0.9)]
        df_test = df.iloc[round(n_points * 0.9):]

        predictor = Predictor(name='test')

        predictor.learn(
            from_data=df_train,
            to_predict='y',
            advanced_args=dict(deduplicate_data=False),
            stop_training_in_x_seconds=40,
            use_gpu=False
        )

        model_data = F.get_model_data('test')
        assert model_data['data_analysis_v2']['tags']['typing']['data_type'] == DATA_TYPES.CATEGORICAL
        assert model_data['data_analysis_v2']['tags']['typing']['data_subtype'] == DATA_SUBTYPES.TAGS

        predictions = predictor.predict(when_data=df_test)
        test_y = df_test.y.apply(str)

        predicted_y = []
        for i in range(len(predictions)):
            predicted_y.append(predictions[i]['y'])

        score = accuracy_score(test_y, predicted_y)
        assert score >= 0.2

    def test_category_tags_output(self):
        vocab = random.sample(SMALL_VOCAB, 10)
        vocab = {i: word for i, word in enumerate(vocab)}
        # x1 contains the index of first tag present
        # x2 contains the index of second tag present
        # if a tag is missing then x1/x2 contain -1 instead
        # Thus the dataset should be perfectly predicted
        n_points = 5000
        x1 = [random.randint(0, len(vocab) - 1) if random.random() > 0.1 else -1 for i in range(n_points)]
        x2 = [random.randint(0, len(vocab) - 1) if random.random() > 0.1 else -1 for i in range(n_points)]
        tags = []
        for x1_index, x2_index in zip(x1, x2):
            row_tags = set([vocab.get(x1_index), vocab.get(x2_index)])
            row_tags = [x for x in row_tags if x is not None]
            tags.append(','.join(row_tags))

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'tags': tags})

        df_train = df.iloc[:round(n_points * 0.9)]
        df_test = df.iloc[round(n_points * 0.9):]

        predictor = Predictor('test')

        predictor.learn(
            from_data=df_train,
            to_predict='tags',
            advanced_args=dict(deduplicate_data=False),
            stop_training_in_x_seconds=60,
            use_gpu=False
        )

        model_data = F.get_model_data('test')
        assert model_data['data_analysis_v2']['tags']['typing']['data_type'] == DATA_TYPES.CATEGORICAL
        assert model_data['data_analysis_v2']['tags']['typing']['data_subtype'] == DATA_SUBTYPES.TAGS

        predictions = predictor.predict(when_data=df_test)
        test_tags = df_test.tags.apply(lambda x: x.split(','))

        predicted_tags = []
        for i in range(len(predictions)):
            predicted_tags.append(predictions[i]['tags'])

        test_tags_encoded = predictor.transaction.model_backend.predictor._mixer.encoders['tags'].encode(test_tags)
        pred_labels_encoded = predictor.transaction.model_backend.predictor._mixer.encoders['tags'].encode(predicted_tags)
        score = f1_score(test_tags_encoded, pred_labels_encoded, average='weighted')

        assert score >= 0.3

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

        model_data = F.get_model_data('test_predictor_deduplicate_data_true')

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

        model_data = F.get_model_data('test_predictor_deduplicate_data_false')

        # Duplicate row was used for analysis and training

        assert model_data['data_preparation']['total_row_count'] == n_points + 1
        assert model_data['data_preparation']['used_row_count'] <= n_points + 1

        assert sum([model_data['data_preparation']['train_row_count'],
                    model_data['data_preparation']['validation_row_count'],
                    model_data['data_preparation']['test_row_count']]) == n_points + 1

        assert sum([predictor.transaction.input_data.train_df.shape[0],
                    predictor.transaction.input_data.test_df.shape[0],
                    predictor.transaction.input_data.validation_df.shape[0]]) == n_points + 1

    def test_empty_column(self):
        mdb = Predictor(name='test_empty_column')

        n_points = 100
        input_dataframe = pd.DataFrame({
            'empty_col': [None] * n_points,
            'numeric_x': list(range(n_points)),
            'categorical_x': [int(x % 2 == 0) for x in range(n_points)],
        }, index=list(range(n_points)))

        input_dataframe['numeric_y'] = input_dataframe.numeric_x + 2 * input_dataframe.categorical_x

        mdb.learn(
            from_data=input_dataframe,
            to_predict='numeric_y',
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={'debug': True}
        )

        mdb.predict(when_data={'categorical_x': 0})

    def test_output_class_distribution(self):
        mdb = Predictor(name='test_output_class_distribution')
        data = load_iris(as_frame=True)
        df = data.data
        df['target'] = data.target

        mdb.learn(
            from_data=df,
            to_predict='target',
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={'debug': True, 'output_class_distribution': True}
        )

        results = mdb.predict(when_data=data.data.iloc[[0]])
        assert 'target_class_distribution' in results._data
        probs = results._data['target_class_distribution']
        assert np.isclose(np.sum(probs), 1)
        for dist in probs:
            assert len(dist) == data.target_names.size
