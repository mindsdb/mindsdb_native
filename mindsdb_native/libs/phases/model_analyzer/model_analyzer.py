from mindsdb_native.libs.helpers.general_helpers import pickle_obj, disable_console_output
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.helpers.general_helpers import evaluate_accuracy
from mindsdb_native.libs.helpers.probabilistic_validator import ProbabilisticValidator
from mindsdb_native.libs.data_types.mindsdb_logger import log
from sklearn.metrics import accuracy_score, r2_score

import pandas as pd
import numpy as np


class ModelAnalyzer(BaseModule):

    def run(self):
        np.seterr(divide='warn', invalid='warn')
        """
        # Runs the model on the validation set in order to fit a probabilistic model that will evaluate the accuracy of future predictions
        """

        output_columns = self.transaction.lmd['predict_columns']
        input_columns = [col for col in self.transaction.lmd['columns'] if col not in output_columns and col not in self.transaction.lmd['columns_to_ignore']]

        # Make predictions on the validation dataset normally and with various columns missing
        normal_predictions = self.transaction.model_backend.predict('validate')


        normal_predictions_test = self.transaction.model_backend.predict('test')
        normal_accuracy = evaluate_accuracy(normal_predictions,
                                            self.transaction.input_data.validation_df,
                                            self.transaction.lmd['stats_v2'],
                                            output_columns,
                                            backend=self.transaction.model_backend)
        
        for col in output_columns:
            reals = self.transaction.input_data.validation_df[col]
            preds = normal_predictions[col]

            fails = False
            
            data_type = self.transaction.lmd['stats_v2'][col]['typing']['data_type']
            data_subtype = self.transaction.lmd['stats_v2'][col]['typing']['data_subtype']
            
            if data_type == DATA_TYPES.CATEGORICAL:
                if data_subtype == DATA_SUBTYPES.TAGS:
                    encoder = self.transaction.model_backend.predictor._mixer.encoders[col]
                    if accuracy_score(encoder.encode(reals), encoder.encode(preds)) <= self.transaction.lmd['stats_v2'][col]['guess_probability']:
                        fails = True
                else:
                    if accuracy_score(reals, preds) <= self.transaction.lmd['stats_v2'][col]['guess_probability']:
                        fails = True
            elif data_type == DATA_TYPES.NUMERIC:
                if r2_score(reals, preds) < 0:
                    fails = True
            else:
                pass

            if fails:
                if not self.transaction.lmd['force_predict']:
                    def predict_wrapper(*args, **kwargs):
                        raise Exception('Failed to train model')
                    self.session.predict = predict_wrapper
                log.error('Failed to train model to predict {}'.format(col))

        empty_input_predictions = {}
        empty_input_accuracy = {}
        empty_input_predictions_test = {}

        ignorable_input_columns = [x for x in input_columns if self.transaction.lmd['stats_v2'][x]['typing']['data_type'] != DATA_TYPES.FILE_PATH
                           and x not in [y[0] for y in self.transaction.lmd['model_order_by']]]

        for col in ignorable_input_columns:
            empty_input_predictions[col] = self.transaction.model_backend.predict('validate', ignore_columns=[col])
            empty_input_predictions_test[col] = self.transaction.model_backend.predict('test', ignore_columns=[col])
            empty_input_accuracy[col] = evaluate_accuracy(empty_input_predictions[col],
                                                          self.transaction.input_data.validation_df,
                                                          self.transaction.lmd['stats_v2'],
                                                          output_columns,
                                                          backend=self.transaction.model_backend)

        # Get some information about the importance of each column
        self.transaction.lmd['column_importances'] = {}
        for col in ignorable_input_columns:
            accuracy_increase = (normal_accuracy - empty_input_accuracy[col])
            # normalize from 0 to 10
            self.transaction.lmd['column_importances'][col] = 10 * max(0, accuracy_increase)

        # Run Probabilistic Validator
        overall_accuracy_arr = []
        self.transaction.lmd['accuracy_histogram'] = {}
        self.transaction.lmd['confusion_matrices'] = {}
        self.transaction.lmd['accuracy_samples'] = {}
        self.transaction.hmd['probabilistic_validators'] = {}


        self.transaction.lmd['train_data_accuracy'] = {}
        self.transaction.lmd['test_data_accuracy'] = {}
        self.transaction.lmd['valid_data_accuracy'] = {}

        for col in output_columns:

            # Training data accuracy
            predictions = self.transaction.model_backend.predict(
                'predict_on_train_data',
                ignore_columns=self.transaction.lmd['stats_v2']['columns_to_ignore']
            )
            self.transaction.lmd['train_data_accuracy'][col] = evaluate_accuracy(
                predictions,
                self.transaction.input_data.train_df,
                self.transaction.lmd['stats_v2'],
                [col],
                backend=self.transaction.model_backend
            )

            # Testing data accuracy
            predictions = self.transaction.model_backend.predict(
                'test',
                ignore_columns=self.transaction.lmd['stats_v2']['columns_to_ignore']
            )
            self.transaction.lmd['test_data_accuracy'][col] = evaluate_accuracy(
                predictions,
                self.transaction.input_data.test_df,
                self.transaction.lmd['stats_v2'],
                [col],
                backend=self.transaction.model_backend
            )

            # Validation data accuracy
            predictions = self.transaction.model_backend.predict(
                'validate',
                ignore_columns=self.transaction.lmd['stats_v2']['columns_to_ignore']
            )
            self.transaction.lmd['valid_data_accuracy'][col] = evaluate_accuracy(
                predictions,
                self.transaction.input_data.validation_df,
                self.transaction.lmd['stats_v2'],
                [col],
                backend=self.transaction.model_backend
            )

        for col in output_columns:
            pval = ProbabilisticValidator(col_stats=self.transaction.lmd['stats_v2'][col], col_name=col, input_columns=input_columns)
            predictions_arr = [normal_predictions_test] + [x for x in empty_input_predictions_test.values()]

            pval.fit(self.transaction.input_data.test_df, predictions_arr, [[ignored_column] for ignored_column in empty_input_predictions_test])
            overall_accuracy, accuracy_histogram, cm, accuracy_samples = pval.get_accuracy_stats()
            overall_accuracy_arr.append(overall_accuracy)

            self.transaction.lmd['accuracy_histogram'][col] = accuracy_histogram
            self.transaction.lmd['confusion_matrices'][col] = cm
            self.transaction.lmd['accuracy_samples'][col] = accuracy_samples
            self.transaction.hmd['probabilistic_validators'][col] = pickle_obj(pval)

        self.transaction.lmd['validation_set_accuracy'] = sum(overall_accuracy_arr)/len(overall_accuracy_arr)
