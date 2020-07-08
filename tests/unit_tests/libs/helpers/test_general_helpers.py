import pandas as pd
from mindsdb_native.libs.constants.mindsdb import DATA_TYPES
from mindsdb_native.libs.helpers.general_helpers import (evaluate_accuracy)


class TestEvaluateAccuracy:
    def test_evaluate_regression(self):
        predictions = {
            'y': [1, 2, 3, 4],
            'y_confidence_range': [
                [0, 2],
                [0, 2],
                [1, 3],
                [4, 4],
            ]
        }

        col_stats = {
            'y': {'typing': {'data_type': DATA_TYPES.NUMERIC}}
        }

        output_columns = ['y']

        data_frame = pd.DataFrame({'y': [1, 2, 3, 5]})

        accuracy = evaluate_accuracy(predictions, data_frame, col_stats, output_columns)

        assert round(accuracy, 2) == 0.75

    def test_evaluate_classification(self):
        predictions = {
            'y': [1, 2, 3, 4]
        }

        col_stats = {
            'y': {'typing': {'data_type': DATA_TYPES.CATEGORICAL}}
        }

        output_columns = ['y']

        data_frame = pd.DataFrame({'y': [1, 2, 3, 5]})

        accuracy = evaluate_accuracy(predictions, data_frame, col_stats, output_columns)

        assert round(accuracy, 2) == 0.75

    def test_evaluate_two_columns(self):
        predictions = {
            'y1': [1, 2, 3, 4],
            'y1_confidence_range': [
                [0, 2],
                [0, 2],
                [1, 3],
                [4, 4],
            ],
            'y2': [0, 0, 1, 1]
        }

        col_stats = {
            'y1': {'typing': {'data_type': DATA_TYPES.NUMERIC}},
            'y2': {'typing': {'data_type': DATA_TYPES.CATEGORICAL}}
        }

        output_columns = ['y1', 'y2']

        data_frame = pd.DataFrame({'y1': [1, 2, 3, 5], 'y2': [1, 0, 1, 0]})

        accuracy = evaluate_accuracy(predictions, data_frame, col_stats, output_columns)

        assert round(accuracy, 2) == round((0.75 + 0.5)/2, 2)

    def test_evaluate_weird_data_types(self):
        for dtype in [
            DATA_TYPES.DATE,
            DATA_TYPES.TEXT,
            DATA_TYPES.SEQUENTIAL,
            DATA_TYPES.FILE_PATH,
        ]:
            predictions = {
                'y': ["1", "2", "3", "4"]
            }

            col_stats = {
                'y': {'typing': {'data_type': dtype}}
            }

            output_columns = ['y']

            data_frame = pd.DataFrame({'y': ["1", "2", "3", "5"]})

            accuracy = evaluate_accuracy(predictions, data_frame, col_stats,
                                         output_columns)

            assert round(accuracy, 2) == 0.75
