import json
import random
from unittest import mock

import pytest
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

from unit_tests.utils import (test_column_types,
                            generate_value_cols,
                            generate_timeseries_labels,
                            generate_log_labels,
                            columns_to_file,
                            PickableMock,
                            SMALL_VOCAB)

from mindsdb_native.libs.helpers.stats_helpers import sample_data


class TestPredictor:
    def test_sample_for_training(self):
        predictor = Predictor(name='test')

        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'categorical_binary': [0, 1] * (n_points // 2),
        }, index=list(range(n_points)))
        input_dataframe['y'] = input_dataframe.numeric_int + input_dataframe.numeric_int*input_dataframe.categorical_binary

        mock_function = PickableMock(spec=sample_data,
                                       wraps=sample_data)
        setattr(mock_function, '__name__', 'mock_sample_data')
        with mock.patch('mindsdb_native.libs.controllers.predictor.sample_data',
            mock_function):

            predictor.learn(from_data=input_dataframe,
                            to_predict='y',
                            backend='lightwood',
                            sample_settings={'sample_for_training': True,
                                             'sample_for_analysis': True},
                            stop_training_in_x_seconds=1,
                            use_gpu=False)

            assert mock_function.called

            # 1 call when sampling for analysis
            # 1 call when sampling training data for lightwood
            # 1 call when sampling testing data for lightwood
            assert mock_function.call_count == 3

    def test_analyze_dataset(self):
        n_points = 100
        n_category_values = 4
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_float': np.linspace(0, n_points, n_points),
            'date_timestamp': [
                (datetime.now() - timedelta(minutes=int(i))).isoformat() for i in
                range(n_points)],
            'date_date': [
                (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in
                range(n_points)],
            'categorical_str': [f'a{x}' for x in (
                    list(range(n_category_values)) * (
                        n_points // n_category_values))],
            'categorical_int': [x for x in (list(range(n_category_values)) * (
                    n_points // n_category_values))],
            'categorical_binary': [0, 1] * (n_points // 2),
            'sequential_array': [f"1,2,3,4,5,{i}" for i in range(n_points)]
        }, index=list(range(n_points)))

        model_data = F.analyse_dataset(from_data=input_dataframe)
        for col, col_data in model_data['data_analysis_v2'].items():
            expected_type = test_column_types[col][0]
            expected_subtype = test_column_types[col][1]
            assert col_data['typing']['data_type'] == expected_type
            assert col_data['typing']['data_subtype'] == expected_subtype

            assert col_data['empty']
            assert col_data['histogram']
            assert 'percentage_buckets' in col_data
            assert 'nr_warnings' in col_data
            assert col_data['identifier'] is None

        assert isinstance(json.dumps(model_data), str)

    def test_sample_for_analysis(self):
        n_points = 100
        n_category_values = 4
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_float': np.linspace(0, n_points, n_points),
            'date_timestamp': [
                (datetime.now() - timedelta(minutes=int(i))).isoformat() for i in
                range(n_points)],
            'date_date': [
                (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in
                range(n_points)],
            'categorical_str': [f'a{x}' for x in (
                list(range(n_category_values)) * (
                n_points // n_category_values))],
            'categorical_int': [x for x in (list(range(n_category_values)) * (
                n_points // n_category_values))],
            'categorical_binary': [0, 1] * (n_points // 2),
            'sequential_array': [f"1,2,3,4,5,{i}" for i in range(n_points)]
        }, index=list(range(n_points)))

        mock_function = PickableMock(spec=sample_data,
                                     wraps=sample_data)
        setattr(mock_function, '__name__', 'mock_sample_data')
        with mock.patch('mindsdb_native.libs.controllers.predictor.sample_data',
                        mock_function):
            model_data = F.analyse_dataset(from_data=input_dataframe,
                                           sample_settings={'sample_for_analysis': True})
            assert mock_function.called

        for col, col_data in model_data['data_analysis_v2'].items():
            expected_type = test_column_types[col][0]
            expected_subtype = test_column_types[col][1]
            assert col_data['typing']['data_type'] == expected_type
            assert col_data['typing']['data_subtype'] == expected_subtype

            assert col_data['empty']
            assert col_data['histogram']
            assert 'percentage_buckets' in col_data
            assert 'nr_warnings' in col_data
            assert col_data['identifier'] is None

    def test_ignore_columns(self):
        input_dataframe = pd.DataFrame({
            'do_use': list(range(100)),
            'y': list(range(100)),
            'ignore_this': list(range(100, 0, -1))
        })

        predictor = Predictor(name='test')
        predictor.learn(
            from_data=input_dataframe,
            to_predict='y',
            ignore_columns=['ignore_this'],
            stop_training_in_x_seconds=1,
            use_gpu=False
        )
        transaction = predictor.transaction

        assert 'do_use' in transaction.input_data.train_df.columns
        assert 'ignore_this' not in transaction.input_data.train_df.columns

    def test_ignore_foreign_keys(self):
        input_dataframe = pd.DataFrame({
            'do_use': list(range(100)),
            'numeric_id': list(range(100)),
            'y': list(range(100)),
        })

        predictor = Predictor(name='test')
        predictor.learn(
            from_data=input_dataframe,
            to_predict='y',
            stop_training_in_x_seconds=1,
            use_gpu=False
        )

        transaction = predictor.transaction

        assert 'do_use' in transaction.input_data.train_df.columns
        # Foreign key is ignored and removed from data frames
        assert 'numeric_id' not in transaction.input_data.train_df.columns
        assert 'numeric_id' in transaction.lmd['columns_to_ignore']

        predictor = Predictor(name='test')
        predictor.learn(
            from_data=input_dataframe,
            to_predict='y',
            stop_training_in_x_seconds=1,
            advanced_args={'force_column_usage': ['numeric_id']},
            use_gpu=False
        )

        transaction = predictor.transaction

        assert 'do_use' in transaction.input_data.train_df.columns
        assert 'numeric_id' in transaction.input_data.train_df.columns
        assert 'numeric_id' not in transaction.lmd['columns_to_ignore']

    def test_analyze_dataset_empty_column(self):
        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'empty_column': [None for i in range(n_points)]
        }, index=list(range(n_points)))

        model_data = F.analyse_dataset(from_data=input_dataframe)

        assert model_data['data_analysis_v2']['empty_column']['empty']['is_empty'] is True

    def test_analyze_dataset_empty_values(self):
        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_int2': list(range(n_points)),
        }, index=list(range(n_points)))
        input_dataframe['numeric_int'].iloc[::2] = None

        model_data = F.analyse_dataset(from_data=input_dataframe)

        assert model_data['data_analysis_v2']['numeric_int']['empty']['empty_percentage'] == 50

    def test_predictor_deduplicate_data(self):
        n_points = 100
        input_dataframe = pd.DataFrame({
            'numeric_int': [x % 44 for x in list(range(n_points))],
            'numeric_int_2': [x % 20 for x in list(range(n_points))],
        }, index=list(range(n_points)))
        input_dataframe['y'] = input_dataframe['numeric_int'] % 10

        # Add duplicate row
        input_dataframe = input_dataframe.append(input_dataframe.iloc[99], ignore_index=True)

        mdb = Predictor(name='test_drop_duplicates')
        mdb.learn(
            from_data=input_dataframe,
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

        assert sum([mdb.transaction.input_data.train_df.shape[0],
                    mdb.transaction.input_data.test_df.shape[0],
                    mdb.transaction.input_data.validation_df.shape[0]]) == n_points

        # Disable deduplication and ensure the duplicate row is used
        mdb = Predictor(name='test_drop_duplicates')
        mdb.learn(
            from_data=input_dataframe,
            to_predict='y',
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={
                'deduplicate_data': False
            }
        )

        model_data = F.get_model_data('test_drop_duplicates')

        # Duplicate row was used for analysis and training

        assert model_data['data_preparation']['total_row_count'] == n_points+1
        assert model_data['data_preparation']['used_row_count'] <= n_points+1

        assert sum([model_data['data_preparation']['train_row_count'],
                    model_data['data_preparation']['validation_row_count'],
                    model_data['data_preparation']['test_row_count']]) == n_points+1

        assert sum([mdb.transaction.input_data.train_df.shape[0],
                    mdb.transaction.input_data.test_df.shape[0],
                    mdb.transaction.input_data.validation_df.shape[0]]) == n_points+1

    @pytest.mark.slow
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
            advanced_args={'force_predict': True}
        )

        result = mdb.predict(when_data={"numeric_x": 10, 'categorical_x': 1})
        explanation_new = result[0].explanation['numeric_y']
        assert isinstance(explanation_new['predicted_value'], int)
        assert isinstance(explanation_new['confidence_interval'],list)
        assert isinstance(explanation_new['confidence_interval'][0],int)
        assert isinstance(explanation_new['important_missing_information'], list)
        assert isinstance(explanation_new['prediction_quality'], str)

        assert len(str(result[0])) > 20

    @pytest.mark.skip(reason="Causes error in probabilistic validator")
    @pytest.mark.slow
    def test_custom_backend(self):
        predictor = Predictor(name='custom_model_test_predictor')

        class CustomDTModel():
            def __init__(self):
                self.clf = LinearRegression()
                le = preprocessing.LabelEncoder()

            def set_transaction(self, transaction):
                self.transaction = transaction
                self.output_columns = self.transaction.lmd['predict_columns']
                self.input_columns = [x for x in self.transaction.lmd['columns']
                                      if x not in self.output_columns]
                self.train_df = self.transaction.input_data.train_df
                self.test_dt = train_df = self.transaction.input_data.test_df

            def train(self):
                self.le_arr = {}
                for col in [*self.output_columns, *self.input_columns]:
                    self.le_arr[col] = preprocessing.LabelEncoder()
                    self.le_arr[col].fit(pd.concat(
                        [self.transaction.input_data.train_df,
                         self.transaction.input_data.test_df,
                         self.transaction.input_data.validation_df])[col])

                X = []
                for col in self.input_columns:
                    X.append(self.le_arr[col].transform(
                        self.transaction.input_data.train_df[col]))

                X = np.swapaxes(X, 1, 0)

                # Only works with one output column
                Y = self.le_arr[self.output_columns[0]].transform(
                    self.transaction.input_data.train_df[self.output_columns[0]])

                self.clf.fit(X, Y)

            def predict(self, mode='predict', ignore_columns=[]):
                if mode == 'predict':
                    df = self.transaction.input_data.data_frame
                if mode == 'validate':
                    df = self.transaction.input_data.validation_df
                elif mode == 'test':
                    df = self.transaction.input_data.test_df

                X = []
                for col in self.input_columns:
                    X.append(self.le_arr[col].transform(df[col]))

                X = np.swapaxes(X, 1, 0)

                predictions = self.clf.predict(X)

                formated_predictions = {self.output_columns[0]: predictions}

                return formated_predictions

        dt_model = CustomDTModel()

        predictor.learn(to_predict='rental_price',
                        from_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
                        backend=dt_model,
                        use_gpu=False)
        predictions = predictor.predict(
            when_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
            backend=dt_model)

        assert predictions

    @pytest.mark.slow
    def test_data_source_setting(self):
        data_url = 'https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/classics/german_credit_data/processed_data/test.csv'
        data_source = FileDS(data_url)
        data_source.set_subtypes({})

        data_source_mod = FileDS(data_url)
        data_source_mod.set_subtypes({'credit_usage': 'Int', 'Average_Credit_Balance': 'Short Text',
             'existing_credits': 'Binary Category'})

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

    @pytest.mark.skip(reason='Test gets stuck during learn call, need investigation')
    @pytest.mark.slow
    def test_timeseries(self, tmp_path):
        ts_hours = 12
        data_len = 120
        train_file_name = os.path.join(str(tmp_path), 'train_data.csv')
        test_file_name = os.path.join(str(tmp_path), 'test_data.csv')

        features = generate_value_cols(['date', 'int'], data_len, ts_hours * 3600)
        labels = [generate_timeseries_labels(features)]

        feature_headers = list(map(lambda col: col[0], features))
        label_headers = list(map(lambda col: col[0], labels))

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
            order_by=feature_headers[0],
            # ,window_size_seconds=ts_hours* 3600 * 1.5
            window_size=3,
            stop_training_in_x_seconds=1,
            use_gpu=False
        )

        results = mdb.predict(when_data=test_file_name, use_gpu=False)

        for row in results:
            expect_columns = [label_headers[0],
                              label_headers[0] + '_confidence']
            for col in expect_columns:
                assert col in row

        models = F.get_models()
        model_data = F.get_model_data(models[0]['name'])
        assert model_data

    def test_multilabel_prediction(self, tmp_path):
        train_file_name = os.path.join(str(tmp_path), 'train_data.csv')
        test_file_name = os.path.join(str(tmp_path), 'test_data.csv')
        data_len = 60

        features = generate_value_cols(['int', 'float', 'int', 'float'], data_len)
        labels = []
        labels.append(generate_log_labels(features))
        labels.append(generate_timeseries_labels(features))

        feature_headers = list(map(lambda col: col[0], features))
        label_headers = list(map(lambda col: col[0], labels))

        # Create the training dataset and save it to a file
        columns_train = list(
            map(lambda col: col[1:int(len(col) * 3 / 4)], features))
        columns_train.extend(
            list(map(lambda col: col[1:int(len(col) * 3 / 4)], labels)))
        columns_to_file(columns_train, train_file_name,
                        headers=[*feature_headers, *label_headers])

        # Create the testing dataset and save it to a file
        columns_test = list(
            map(lambda col: col[int(len(col) * 3 / 4):], features))
        columns_to_file(columns_test, test_file_name,
                        headers=feature_headers)

        mdb = Predictor(name='test_multilabel_prediction')
        mdb.learn(
            from_data=train_file_name,
            to_predict=label_headers,
            stop_training_in_x_seconds=1,
            use_gpu=False,
            advanced_args={'force_predict': True}
        )

        results = mdb.predict(when_data=test_file_name)
        models = F.get_models()
        model_data = F.get_model_data(models[0]['name'])
        assert model_data

        for i in range(len(results)):
            row = results[i]
            for label in label_headers:
                expect_columns = [label, label + '_confidence']
                for col in expect_columns:
                    assert col in row

    # If cuda is not available then we expect the test to fail when trying to use it
    @pytest.mark.parametrize("use_gpu", [
        True if torch.cuda.is_available() else pytest.param(True, marks=pytest.mark.xfail),
        False])
    @pytest.mark.slow
    def test_house_pricing(self, use_gpu):
        """
        Tests whole pipeline from downloading the dataset to making predictions and explanations.
        """
        # Create & Learn
        mdb = Predictor(name='home_rentals_price')
        mdb.learn(to_predict='rental_price',
                  from_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
                  backend='lightwood',
                  stop_training_in_x_seconds=180,
                  use_gpu=use_gpu)

        def assert_prediction_interface(predictions):
            for prediction in predictions:
                assert hasattr(prediction, 'explanation')

        test_results = mdb.test(
            when_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
            accuracy_score_functions=r2_score, predict_args={'use_gpu': use_gpu})
        assert test_results['rental_price_accuracy'] >= 0.8

        predictions = mdb.predict(
            when_data="https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv",
            use_gpu=use_gpu)
        assert_prediction_interface(predictions)
        predictions = mdb.predict(when_data={'sqft': 300}, use_gpu=use_gpu)
        assert_prediction_interface(predictions)

        amd = F.get_model_data('home_rentals_price')
        assert isinstance(json.dumps(amd), str)

        for k in ['status', 'name', 'version', 'data_source', 'current_phase',
                  'updated_at', 'created_at',
                  'train_end_at']:
            assert isinstance(amd[k], str)

        assert isinstance(amd['predict'], (list, str))
        assert isinstance(amd['is_active'], bool)

        for k in ['validation_set_accuracy', 'accuracy']:
            assert isinstance(amd[k], float)

        for k in amd['data_preparation']:
            assert isinstance(amd['data_preparation'][k], (int, float))

        for k in amd['data_analysis']:
            assert (len(amd['data_analysis'][k]) > 0)
            assert isinstance(amd['data_analysis'][k][0], dict)

        model_analysis = amd['model_analysis']
        assert (len(model_analysis) > 0)
        assert isinstance(model_analysis[0], dict)
        input_importance = model_analysis[0]["overall_input_importance"]
        assert (len(input_importance) > 0)
        assert isinstance(input_importance, dict)

        for k in ['train', 'test', 'valid']:
            assert isinstance(model_analysis[0][k + '_data_accuracy'], dict)
            assert len(model_analysis[0][k + '_data_accuracy']) == 1
            assert model_analysis[0][k + '_data_accuracy']['rental_price'] > 0.60

        for column, importance in zip(input_importance["x"],
                                      input_importance["y"]):
            assert isinstance(column, str)
            assert (len(column) > 0)
            assert isinstance(importance, (float, int))
            assert (importance >= 0 and importance <= 10)
            
    @pytest.mark.slow
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

        predictor.learn(from_data=df_train, to_predict='y',
                        advanced_args=dict(deduplicate_data=False),
                        stop_training_in_x_seconds=60,
                        use_gpu=False)

        model_data = F.get_model_data('test')
        assert model_data['data_analysis_v2']['tags']['typing']['data_type'] == DATA_TYPES.CATEGORICAL
        assert model_data['data_analysis_v2']['tags']['typing']['data_subtype'] == DATA_SUBTYPES.TAGS

        predictions = predictor.predict(when_data=df_test)
        test_y = df_test.y.apply(str)

        predicted_y = []
        for i in range(len(predictions)):
            predicted_y.append(predictions[i]['y'])

        score = accuracy_score(test_y, predicted_y)
        assert score >= 0.3

    @pytest.mark.slow
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

        predictor.learn(from_data=df_train,
                        to_predict='tags',
                        advanced_args=dict(deduplicate_data=False),
                        stop_training_in_x_seconds=60,
                        use_gpu=False)

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

    def test_user_provided_split_indices(self):
        predictor = Predictor('test')

        df = pd.DataFrame({
            'col_a': [*range(100)],
            'col_b': [*range(100)]
        })

        predictor.learn(
            from_data=df,
            to_predict='col_b',
            advanced_args={
                'data_split_indexes': {
                    'validation_indexes': [*range(0, 30)],
                    'train_indexes': [*range(30, 60)],
                    'test_indexes': [*range(60, 100)]
                }
            },
            use_gpu=False
        )

        assert set(predictor.transaction.input_data.train_df['col_a'].tolist()) == set(range(30, 60))
        assert set(predictor.transaction.input_data.test_df['col_a'].tolist()) == set(range(60, 100))
        assert set(predictor.transaction.input_data.validation_df['col_a'].tolist()) == set(range(0, 30))
