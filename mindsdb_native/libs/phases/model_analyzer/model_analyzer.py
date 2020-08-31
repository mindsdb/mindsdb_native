from mindsdb_native.libs.helpers.general_helpers import pickle_obj, disable_console_output
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.helpers.general_helpers import evaluate_accuracy
from mindsdb_native.libs.helpers.conformal_helpers import ConformalClassifierAdapter, ConformalRegressorAdapter
from mindsdb_native.libs.helpers.probabilistic_validator import ProbabilisticValidator

import numpy as np
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from nonconformist.icp import IcpRegressor, IcpClassifier
from nonconformist.nc import RegressorNc, AbsErrorErrFunc, ClassifierNc, MarginErrFunc


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

        # conformal prediction confidence estimation
        target = self.transaction.lmd['predict_columns'][0]
        X = self.transaction.input_data.train_df.copy(deep=True)
        y = X.pop(target)

        data_type = self.transaction.lmd['stats_v2'][self.transaction.lmd['predict_columns'][0]]['typing']['data_type']
        is_classification = data_type == DATA_TYPES.CATEGORICAL

        # TODO: what if n_target > 1?
        fit_params = {'target': target,
                      'all_columns': self.transaction.lmd['columns'],
                      'columns_to_ignore': self.transaction.lmd['columns_to_ignore']}

        if not is_classification:
            self.transaction.lmd['stats_v2']['train_std_dev'] = self.transaction.input_data.train_df[target].std()


        if is_classification:
            enc = OneHotEncoder(sparse=False)
            enc.fit(y.values.reshape(-1, 1))
            fit_params['one_hot_enc'] = enc
            self.transaction.lmd['label_encoder'] = enc

            adapter = ConformalClassifierAdapter
            nc_function = MarginErrFunc()  # better than IPS as we'd need the complete distribution over all classes
            nc_class = ClassifierNc
            icp_class = IcpClassifier
        else:
            adapter = ConformalRegressorAdapter
            nc_function = AbsErrorErrFunc()
            nc_class = RegressorNc
            icp_class = IcpRegressor

        model = adapter(self.transaction.model_backend.predictor, fit_params=fit_params)
        nc = nc_class(model, nc_function)
        if is_classification:
            self.transaction.hmd['icp'] = icp_class(nc, smoothing=False) # ?
        else:
            self.transaction.hmd['icp'] = icp_class(nc)
        self.transaction.hmd['icp'].fit(X.values, y.values)

        # calibrate conformal estimator on test set
        X = deepcopy(self.transaction.input_data.test_df)
        y = X.pop(target).values
        if is_classification:
            if isinstance(enc.categories_[0][0], str):
                cats = enc.categories_[0].tolist()
                y = np.array([cats.index(i) for i in y])
            y = y.astype(int)

        self.transaction.hmd['icp'].calibrate(X.values, y)
