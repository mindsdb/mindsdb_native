import os
import uuid
import traceback
import pickle
from pathlib import Path

from mindsdb_native.libs.data_types.mindsdb_logger import MindsdbLogger
from mindsdb_native.libs.helpers.multi_data_source import getDS
from mindsdb_native.__about__ import __version__

from mindsdb_native.config import CONFIG
from mindsdb_native.libs.controllers.transaction import Transaction
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.helpers.general_helpers import check_for_updates, deprecated
from mindsdb_native.libs.controllers.functional import (export_storage, export_predictor,
                                                 rename_model, delete_model,
                                                 import_model, get_model_data, get_models)

class Predictor:

    def __init__(self, name, log_level=CONFIG.DEFAULT_LOG_LEVEL,
                              root_folder=CONFIG.MINDSDB_STORAGE_PATH, # This param is unused, kept for backwards compat
                 ):
        """
        This controller defines the API to a MindsDB 'mind', a mind is an object that can learn and predict from data

        :param name: the namespace you want to identify this mind instance with
        :param root_folder: the folder where you want to store this mind or load from
        :param log_level: the desired log level
        """

        self.name = name
        self.uuid = str(uuid.uuid1())
        self.log = MindsdbLogger(log_level=log_level, uuid=self.uuid)

        if CONFIG.CHECK_FOR_UPDATES:
            check_for_updates()

        if not CONFIG.SAGEMAKER:
            # If storage path is not writable, raise an exception as this can no longer be
            if not os.access(CONFIG.MINDSDB_STORAGE_PATH, os.W_OK):
                error_message = '''Cannot write into storage path, please either set the config variable mindsdb.config.set('MINDSDB_STORAGE_PATH',<path>) or give write access to {folder}'''
                self.log.warning(error_message.format(folder=CONFIG.MINDSDB_STORAGE_PATH))
                raise ValueError(error_message.format(folder=CONFIG.MINDSDB_STORAGE_PATH))


            # If storage path is not writable, raise an exception as this can no longer be
            if not os.access(CONFIG.MINDSDB_STORAGE_PATH, os.R_OK):
                error_message = '''Cannot read from storage path, please either set the config variable mindsdb.config.set('MINDSDB_STORAGE_PATH',<path>) or give write access to {folder}'''
                self.log.warning(error_message.format(folder=CONFIG.MINDSDB_STORAGE_PATH))
                raise ValueError(error_message.format(folder=CONFIG.MINDSDB_STORAGE_PATH))

    @deprecated(reason='Use functional.get_models instead')
    def get_models(self):
        return get_models()

    @deprecated(reason='Use functional.get_model_data instead')
    def get_model_data(self, model_name=None, lmd=None):
        if model_name is None:
            model_name = self.name
        return get_model_data(model_name)

    @deprecated(reason='Use functional.export_storage instead')
    def export(self, mindsdb_storage_dir='mindsdb_storage'):
        try:
            export_storage(mindsdb_storage_dir)
            return True
        except Exception:
            return False

    @deprecated(reason='Use functional.export_predictor instead')
    def export_model(self, model_name=None):
        """
        If you want to export a model to a file

        :param model_name: this is the name of the model you wish to export (defaults to the name of the current Predictor)
        :return: bool (True/False) True if mind was exported successfully
        """
        if not model_name:
            model_name = self.name
        try:
            export_predictor(model_name)
            return True
        except Exception:
            return False

    @deprecated(reason='Use functional.import_model instead')
    def load(self, model_archive_path):
        return import_model(model_archive_path)

    @deprecated(reason='Use functional.import_model instead')
    def load_model(self, model_archive_path):
        return import_model(model_archive_path)

    @deprecated(reason='Use functional.rename_model instead')
    def rename_model(self, old_model_name, new_model_name):
        try:
            return rename_model(old_model_name, new_model_name)
        except Exception:
            return False

    @deprecated(reason='Use functional.delete_model instead')
    def delete_model(self, model_name=None):
        if not model_name:
            model_name = self.name
        try:
            delete_model(model_name)
            return True
        except Exception as e:
            return False

    def analyse_dataset(self, from_data, sample_margin_of_error=0.005):
        """
        Analyse the particular dataset being given
        """

        from_ds = getDS(from_data)
        transaction_type = TRANSACTION_ANALYSE
        sample_confidence_level = 1 - sample_margin_of_error

        heavy_transaction_metadata = dict(
            name = self.name,
            from_data = from_ds
        )

        light_transaction_metadata = dict(
            version = str(__version__),
            name = self.name,
            model_columns_map = from_ds._col_map,
            type = transaction_type,
            sample_margin_of_error = sample_margin_of_error,
            sample_confidence_level = sample_confidence_level,
            model_is_time_series = False,
            model_group_by = [],
            model_order_by = [],
            columns_to_ignore = [],
            data_preparation = {},
            predict_columns = [],
            empty_columns = [],
            handle_foreign_keys = True,
            force_categorical_encoding = [],
            handle_text_as_categorical = False,
            data_types = {},
            data_subtypes = {}
        )

        Transaction(session=self, light_transaction_metadata=light_transaction_metadata, heavy_transaction_metadata=heavy_transaction_metadata, logger=self.log)
        return get_model_data(model_name=None, lmd=light_transaction_metadata)


    def learn(self,
              to_predict,
              from_data,
              test_from_data=None,
              group_by=None,
              window_size=None,
              order_by=None,
              sample_margin_of_error=0.005,
              ignore_columns=None,
              stop_training_in_x_seconds=None,
              backend='lightwood',
              rebuild_model=True,
              use_gpu=None,
              disable_optional_analysis=False,
              equal_accuracy_for_all_output_categories=True,
              output_categories_importance_dictionary=None,
              unstable_parameters_dict=None):
        """
        Learn to predict a column or columns from the data in 'from_data'

        Mandatory arguments:
        :param to_predict: what column or columns you want to predict
        :param from_data: the data that you want to learn from, this can be either a file, a pandas data frame, or url or a mindsdb data source

        Optional arguments:
        :param test_from_data: If you would like to test this learning from a different data set

        Optional Time series arguments:
        :param order_by: this order by defines the time series, it can be a list. By default it sorts each sort by column in ascending manner, if you want to change this pass a touple ('column_name', 'boolean_for_ascending <default=true>')
        :param group_by: This argument tells the time series that it should learn by grouping rows by a given id
        :param window_size: The number of samples to learn from in the time series

        Optional data transformation arguments:
        :param ignore_columns: mindsdb will ignore this column

        Optional sampling parameters:
        :param sample_margin_of_error (DEFAULT 0): Maximum expected difference between the true population parameter, such as the mean, and the sample estimate.

        Optional debug arguments:
        :param stop_training_in_x_seconds: (default None), if set, you want training to finish in a given number of seconds

        :return:
        """

        if ignore_columns is None:
            ignore_columns = []

        if group_by is None:
            group_by = []

        if order_by is None:
            order_by = []

        # lets turn into lists: predict, ignore, group_by, order_by
        predict_columns = to_predict if isinstance(to_predict, list) else [to_predict]
        ignore_columns = ignore_columns if isinstance(ignore_columns, list) else [ignore_columns]
        group_by = group_by if isinstance(group_by, list) else [group_by]
        order_by = order_by if isinstance(order_by, list) else [order_by]

        # lets turn order by into list of tuples if not already
        # each element ('column_name', 'boolean_for_ascending <default=true>')
        order_by = [col_name if isinstance(col_name, tuple) else (col_name, True) for col_name in order_by]

        if unstable_parameters_dict is None:
            unstable_parameters_dict = {}

        from_ds = getDS(from_data)

        test_from_ds = None if test_from_data is None else getDS(test_from_data)

        transaction_type = TRANSACTION_LEARN
        sample_confidence_level = 1 - sample_margin_of_error

        if len(predict_columns) == 0:
            error = 'You need to specify a column to predict'
            self.log.error(error)
            raise ValueError(error)

        is_time_series = True if len(order_by) > 0 else False

        """
        We don't implement "name" as a concept in mindsdbd data sources, this is only available for files,
        the server doesn't handle non-file data sources at the moment, so this shouldn't prove an issue,
        once we want to support datasources such as s3 and databases for the server we need to add name as a concept (or, preferably, before that)
        """
        data_source_name = from_data if isinstance(from_data, str) else 'Unkown'

        heavy_transaction_metadata = dict(
            name=self.name,
            from_data=from_ds,
            test_from_data=test_from_ds,
            bucketing_algorithms = {},
            predictions= None,
            model_backend= backend
        )

        light_transaction_metadata = dict(
            version = str(__version__),
            name = self.name,
            data_preparation = {},
            predict_columns = predict_columns,
            model_columns_map = from_ds._col_map,
            model_group_by = group_by,
            model_order_by = order_by,
            model_is_time_series = is_time_series,
            data_source = data_source_name,
            type = transaction_type,
            window_size = window_size,
            sample_margin_of_error = sample_margin_of_error,
            sample_confidence_level = sample_confidence_level,
            stop_training_in_x_seconds = stop_training_in_x_seconds,
            rebuild_model = rebuild_model,
            model_accuracy = {'train': {}, 'test': {}},
            column_importances = None,
            columns_buckets_importances = None,
            columnless_prediction_distribution = None,
            all_columns_prediction_distribution = None,
            use_gpu = use_gpu,
            columns_to_ignore = ignore_columns,
            disable_optional_analysis = disable_optional_analysis,
            validation_set_accuracy = None,
            lightwood_data = {},
            ludwig_data = {},
            weight_map = {},
            confusion_matrices = {},
            empty_columns = [],
            data_types = {},
            data_subtypes = {},
            equal_accuracy_for_all_output_categories = equal_accuracy_for_all_output_categories,
            output_categories_importance_dictionary = output_categories_importance_dictionary if output_categories_importance_dictionary is not None else {},

            skip_model_training = unstable_parameters_dict.get('skip_model_training', False),
            skip_stats_generation = unstable_parameters_dict.get('skip_stats_generation', False),
            optimize_model = unstable_parameters_dict.get('optimize_model', False),
            force_disable_cache = unstable_parameters_dict.get('force_disable_cache', False),
            force_categorical_encoding = unstable_parameters_dict.get('force_categorical_encoding', []),
            handle_foreign_keys = unstable_parameters_dict.get('handle_foreign_keys', False),
            handle_text_as_categorical = unstable_parameters_dict.get('handle_text_as_categorical', False),
            use_selfaware_model = unstable_parameters_dict.get('use_selfaware_model', True)
        )

        if rebuild_model is False:
            old_lmd = {}
            for k in light_transaction_metadata: old_lmd[k] = light_transaction_metadata[k]

            old_hmd = {}
            for k in heavy_transaction_metadata: old_hmd[k] = heavy_transaction_metadata[k]

            with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, light_transaction_metadata['name'] + '_light_model_metadata.pickle'), 'rb') as fp:
                light_transaction_metadata = pickle.load(fp)

            with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, heavy_transaction_metadata['name'] + '_heavy_model_metadata.pickle'), 'rb') as fp:
                heavy_transaction_metadata= pickle.load(fp)

            for k in ['data_preparation', 'rebuild_model', 'data_source', 'type', 'columns_to_ignore', 'sample_margin_of_error', 'sample_confidence_level', 'stop_training_in_x_seconds']:
                if old_lmd[k] is not None: light_transaction_metadata[k] = old_lmd[k]

            for k in ['from_data', 'test_from_data']:
                if old_hmd[k] is not None: heavy_transaction_metadata[k] = old_hmd[k]
        Transaction(session=self,
                    light_transaction_metadata=light_transaction_metadata,
                    heavy_transaction_metadata=heavy_transaction_metadata,
                    logger=self.log)

    def test(self, when_data, accuracy_score_functions, score_using='predicted_value', predict_args=None):
        """
        :param when_data: use this when you have data in either a file, a pandas data frame, or url to a file that you want to predict from
        :param accuracy_score_functions: a single function or  a dictionary for the form `{f'{target_name}': acc_func}` for when we have multiple targets
        :param score_using: what values from the `explanation` of the target to use in the score function, defaults to the
        :param predict_args: dictionary of arguments to be passed to `predict`, e.g: `predict_args={'use_gpu': True}`

        :return: a dictionary for the form `{f'{target_name}_accuracy': accuracy_func_return}`, e.g. {'rental_price_accuracy':0.99}
        """
        if predict_args is None:
            predict_args = {}

        predictions = self.predict(when_data=when_data, **predict_args)

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, f'{self.name}_light_model_metadata.pickle'), 'rb') as fp:
            lmd = pickle.load(fp)

        accuracy_dict = {}
        for col in lmd['predict_columns']:
            if isinstance(accuracy_score_functions, dict):
                acc_f = accuracy_score_functions[col]
            else:
                acc_f = accuracy_score_functions

            accuracy_dict[f'{col}_accuracy'] = acc_f([x[f'__observed_{col}'] for x in predictions], [x.explanation[col][score_using] for x in predictions])

        return accuracy_dict


    def predict(self,
                when=None,
                when_data=None,
                use_gpu=None,
                unstable_parameters_dict=None,
                backend=None,
                run_confidence_variation_analysis=False):
        """
        You have a mind trained already and you want to make a prediction

        :param when: use this if you have certain conditions for a single prediction
        :param when_data: use this when you have data in either a file, a pandas data frame, or url to a file that you want to predict from
        :param run_confidence_variation_analysis: Run a confidence variation analysis on each of the given input column, currently only works when making single predictions via `when`

        :return: TransactionOutputData object
        """

        if unstable_parameters_dict is None:
            unstable_parameters_dict = {}

        if run_confidence_variation_analysis is True and when_data is not None:
            error_msg = 'run_confidence_variation_analysis=True is a valid option only when predicting a single data point via `when`'
            self.log.error(error_msg)
            raise ValueError(error_msg)

        transaction_type = TRANSACTION_PREDICT
        when_ds = None if when_data is None else getDS(when_data)

        # lets turn into lists: when
        when = [when] if isinstance(when, dict) else when if when is not None else []

        heavy_transaction_metadata = {}
        if when_ds is None:
            heavy_transaction_metadata['when_data'] = None
        else:
            heavy_transaction_metadata['when_data'] = when_ds
        heavy_transaction_metadata['model_when_conditions'] = when
        heavy_transaction_metadata['name'] = self.name

        if backend is not None:
            heavy_transaction_metadata['model_backend'] = backend

        light_transaction_metadata = dict(
            name = self.name,
            type = transaction_type,
            use_gpu = use_gpu,
            data_preparation = {},
            run_confidence_variation_analysis = run_confidence_variation_analysis,
            force_disable_cache = unstable_parameters_dict.get('force_disable_cache', False)
        )

        transaction = Transaction(session=self,
                                  light_transaction_metadata=light_transaction_metadata,
                                  heavy_transaction_metadata=heavy_transaction_metadata)

        return transaction.output_data
