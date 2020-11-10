import os
import sys
import psutil
import uuid
import pickle

from mindsdb_native.__about__ import __version__
from mindsdb_native.libs.data_types.mindsdb_logger import MindsdbLogger
from mindsdb_native.libs.helpers.multi_data_source import getDS
from mindsdb_native.config import CONFIG
from mindsdb_native.libs.controllers.transaction import (
    LearnTransaction, PredictTransaction
)
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.helpers.general_helpers import check_for_updates, load_lmd, load_hmd
from mindsdb_native.libs.helpers.locking import MDBLock
from mindsdb_native.libs.helpers.stats_helpers import sample_data


def _get_memory_optimizations(df):
    df_memory = sys.getsizeof(df)
    total_memory = psutil.virtual_memory().total

    mem_usage_ratio = df_memory / total_memory

    sample_for_analysis = mem_usage_ratio >= 0.1 or (df.shape[0] * df.shape[1]) > (3 * pow(10, 4))
    sample_for_training = mem_usage_ratio >= 0.5
    disable_lightwood_transform_cache = mem_usage_ratio >= 0.2

    return sample_for_analysis, sample_for_training, disable_lightwood_transform_cache


def _prepare_sample_settings(user_provided_settings,
                             sample_for_analysis,
                             sample_for_training):
    default_sample_settings = dict(
        sample_for_analysis=sample_for_analysis,
        sample_for_training=sample_for_training,
        sample_margin_of_error=0.005,
        sample_confidence_level=1 - 0.005,
        sample_percentage=None,
        sample_function=sample_data
    )

    if user_provided_settings:
        default_sample_settings.update(user_provided_settings)
    sample_settings = default_sample_settings

    sample_function = sample_settings['sample_function']

    # We need the settings to be JSON serializable, so the actual function will be stored in heavy metadata
    sample_settings['sample_function'] = sample_settings['sample_function'].__name__

    return sample_settings, sample_function

def _prepare_timeseries_settings(user_provided_settings):
    timeseries_settings = dict(
        is_timeseries=False
        ,group_by=None
        ,order_by=None
        ,window=None
        ,use_previous_target=True
        ,nr_predictions=1
        ,historical_columns=[]
    )

    if len(user_provided_settings) > 0:
        if 'order_by' not in user_provided_settings:
            raise Exception('Invalid timeseries settings, please provide `order_by` key [a list of columns]')
        elif 'window' not in user_provided_settings:
            raise Exception(f'Invalid timeseries settings, you must specify a window size')
        else:
            timeseries_settings['is_timeseries'] = True


    for k in user_provided_settings:
        if k in timeseries_settings:
            timeseries_settings[k] = user_provided_settings[k]
        else:
            raise Exception(f'Invalid timeseries setting: {k}')

    return timeseries_settings


class Predictor:
    def __init__(self, name, log_level=CONFIG.DEFAULT_LOG_LEVEL):
        """
        This controller defines the API to a MindsDB 'mind', a mind is an object that can learn and predict from data

        :param name: the namespace you want to identify this mind instance with
        :param root_folder: the folder where you want to store this mind or load from
        :param log_level: the desired log level
        """
        self.name = name
        self.uuid = str(uuid.uuid1())
        self.log = MindsdbLogger(log_level=log_level, uuid=self.uuid)
        self.breakpoint = None
        self.transaction = None

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

    def quick_learn(self,
              to_predict,
              from_data,
              timeseries_settings=None,
              ignore_columns=None,
              stop_training_in_x_seconds=None,
              backend='lightwood',
              rebuild_model=True,
              use_gpu=None,
              equal_accuracy_for_all_output_categories=True,
              output_categories_importance_dictionary=None,
              advanced_args=None,
              sample_settings=None):

        if advanced_args is None:
            advanced_args = {}
        advanced_args['quick_learn'] = True

        return self.learn(to_predict, from_data, timeseries_settings, ignore_columns, stop_training_in_x_seconds, backend, rebuild_model, use_gpu, equal_accuracy_for_all_output_categories, output_categories_importance_dictionary, advanced_args, sample_settings)


    def learn(self,
              to_predict,
              from_data,
              timeseries_settings=None,
              ignore_columns=None,
              stop_training_in_x_seconds=None,
              backend='lightwood',
              rebuild_model=True,
              use_gpu=None,
              equal_accuracy_for_all_output_categories=True,
              output_categories_importance_dictionary=None,
              advanced_args=None,
              sample_settings=None):
        """
        Learn to predict a column or columns from the data in 'from_data'

        Mandatory arguments:
        :param to_predict: what column or columns you want to predict
        :param from_data: the data that you want to learn from, this can be either a file, a pandas data frame, or url or a mindsdb data source

        Optional Time series arguments:
        :param timeseries_settings: dictionary of options for handling the data as a timeseries

        Optional data transformation arguments:
        :param ignore_columns: mindsdb will ignore this column

        Optional sampling parameters:
        :param sample_settings: dictionary of options for sampling from dataset.
            Includes `sample_for_analysis`. `sample_for_training`, `sample_margin_of_error`, `sample_confidence_level`, `sample_percentage`, `sample_function`.
            Default values depend on the size of input dataset and available memory.
            Generally, the bigger the dataset, the more sampling is used.

        Optional debug arguments:
        :param stop_training_in_x_seconds: (default None), if set, you want training to finish in a given number of seconds

        :return:
        """
        with MDBLock('exclusive', 'learn_' + self.name):
            ignore_columns = [] if ignore_columns is None else ignore_columns
            timeseries_settings = {} if timeseries_settings is None else timeseries_settings
            advanced_args = {} if advanced_args is None else advanced_args

            predict_columns = to_predict if isinstance(to_predict, list) else [to_predict]
            ignore_columns = ignore_columns if isinstance(ignore_columns, list) else [ignore_columns]
            if len(predict_columns) == 0:
                error = 'You need to specify the column[s] you want to predict via the `to_predict` argument!'
                self.log.error(error)
                raise ValueError(error)

            from_ds = getDS(from_data)

            # Set user-provided subtypes
            from_ds.set_subtypes(
                advanced_args.get('subtypes', {})
            )

            transaction_type = TRANSACTION_LEARN

            sample_for_analysis, sample_for_training, disable_lightwood_transform_cache = _get_memory_optimizations(from_ds.df)
            sample_settings, sample_function = _prepare_sample_settings(
                sample_settings,
                sample_for_analysis,
                sample_for_training
            )

            timeseries_settings = _prepare_timeseries_settings(timeseries_settings)

            self.log.warning(f'Sample for analysis: {sample_for_analysis}')
            self.log.warning(f'Sample for training: {sample_for_training}')

            """
            We don't implement "name" as a concept in mindsdbd data sources, this is only available for files,
            the server doesn't handle non-file data sources at the moment, so this shouldn't prove an issue,
            once we want to support datasources such as s3 and databases for the server we need to add name as a concept (or, preferably, before that)
            """
            data_source_name = from_ds.name()

            heavy_transaction_metadata = dict(
                name=self.name,
                from_data=from_ds,
                predictions= None,
                model_backend= backend,
                sample_function=sample_function,
                from_data_type=type(from_ds),
                breakpoint = self.breakpoint
            )

            light_transaction_metadata = dict(
                version = str(__version__),
                name = self.name,
                data_preparation = {},
                predict_columns = predict_columns,
                model_columns_map = from_ds._col_map,
                tss=timeseries_settings,
                data_source = data_source_name,
                type = transaction_type,
                sample_settings = sample_settings,
                stop_training_in_x_seconds = stop_training_in_x_seconds,
                rebuild_model = rebuild_model,
                model_accuracy = {'train': {}, 'test': {}},
                column_importances = None,
                columns_buckets_importances = None,
                columnless_prediction_distribution = None,
                all_columns_prediction_distribution = None,
                use_gpu = use_gpu,
                columns_to_ignore = ignore_columns,
                validation_set_accuracy = None,
                lightwood_data = {},
                weight_map = {},
                confusion_matrices = {},
                empty_columns = [],
                data_types = {},
                data_subtypes = {},
                equal_accuracy_for_all_output_categories = equal_accuracy_for_all_output_categories,
                output_categories_importance_dictionary = output_categories_importance_dictionary if output_categories_importance_dictionary is not None else {},

                force_disable_cache = advanced_args.get('force_disable_cache', disable_lightwood_transform_cache),
                force_categorical_encoding = advanced_args.get('force_categorical_encoding', []),
                force_column_usage = advanced_args.get('force_column_usage', []),
                use_selfaware_model = advanced_args.get('use_selfaware_model', True),
                deduplicate_data = advanced_args.get('deduplicate_data', True),
                null_values = advanced_args.get('null_values', {}),
                data_split_indexes = advanced_args.get('data_split_indexes', None),
                tags_delimiter = advanced_args.get('tags_delimiter', ','),
                force_predict = advanced_args.get('force_predict', False),
                mixer_class = advanced_args.get('use_mixers', None),
                setup_args = from_data.setup_args if hasattr(from_data, 'setup_args') else None,
                debug = advanced_args.get('debug', False),
                allow_incomplete_history = advanced_args.get('allow_incomplete_history', False),
                quick_learn = advanced_args.get('quick_learn', False),
                quick_predict = advanced_args.get('quick_predict', False)
            )

            if rebuild_model is False:
                old_lmd = {}
                for k in light_transaction_metadata: old_lmd[k] = light_transaction_metadata[k]

                old_hmd = {}
                for k in heavy_transaction_metadata: old_hmd[k] = heavy_transaction_metadata[k]

                light_transaction_metadata = load_lmd(os.path.join(
                    CONFIG.MINDSDB_STORAGE_PATH,
                    light_transaction_metadata['name'],
                    'light_model_metadata.pickle'
                ))

                heavy_transaction_metadata = load_hmd(os.path.join(
                    CONFIG.MINDSDB_STORAGE_PATH,
                    heavy_transaction_metadata['name'],
                    'heavy_model_metadata.pickle'
                ))

                for k in ['data_preparation', 'rebuild_model', 'data_source', 'type', 'columns_to_ignore', 'sample_margin_of_error', 'sample_confidence_level', 'stop_training_in_x_seconds']:
                    if old_lmd[k] is not None: light_transaction_metadata[k] = old_lmd[k]

                if old_hmd['from_data'] is not None:
                    heavy_transaction_metadata['from_data'] = old_hmd['from_data']

            self.transaction = LearnTransaction(
                session=self,
                light_transaction_metadata=light_transaction_metadata,
                heavy_transaction_metadata=heavy_transaction_metadata,
                logger=self.log
            )

            self.transaction.run()


    def test(self, when_data, accuracy_score_functions, score_using='predicted_value', predict_args=None):
        """
        :param when_data: use this when you have data in either a file, a pandas data frame, or url to a file that you want to predict from
        :param accuracy_score_functions: a single function or  a dictionary for the form `{f'{target_name}': acc_func}` for when we have multiple targets
        :param score_using: what values from the `explanation` of the target to use in the score function, defaults to the
        :param predict_args: dictionary of arguments to be passed to `predict`, e.g: `predict_args={'use_gpu': True}`

        :return: a dictionary for the form `{f'{target_name}_accuracy': accuracy_func_return}`, e.g. {'rental_price_accuracy':0.99}
        """
        with MDBLock('shared', 'learn_' + self.name):
            if predict_args is None:
                predict_args = {}

            predictions = self.predict(when_data=when_data, **predict_args)

            lmd = load_lmd(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.name, 'light_model_metadata.pickle'))

            accuracy_dict = {}
            for col in lmd['predict_columns']:
                if isinstance(accuracy_score_functions, dict):
                    acc_f = accuracy_score_functions[col]
                else:
                    acc_f = accuracy_score_functions

                if score_using is None:
                    predicted = [x.explanation[col] for x in predictions]
                else:
                    predicted = [x.explanation[col][score_using] for x in predictions]

                real = [x[f'__observed_{col}'] for x in predictions]
                accuracy_dict[f'{col}_accuracy'] = acc_f(real, predicted)

            return accuracy_dict

    def quick_predict(self,
                when_data,
                use_gpu=None,
                advanced_args=None,
                backend=None):

        if advanced_args is None:
            advanced_args = {}
        advanced_args['quick_predict'] = True

        return self.predict(when_data, use_gpu, advanced_args, backend)

    def predict(self,
                when_data,
                use_gpu=None,
                advanced_args=None,
                backend=None):
        """
        You have a mind trained already and you want to make a prediction

        :param when_data: python dict, file path, a pandas data frame, or url to a file that you want to predict from

        :return: TransactionOutputData object
        """
        with MDBLock('shared', 'learn_' + self.name):
            if advanced_args is None:
                advanced_args = {}

            transaction_type = TRANSACTION_PREDICT
            when_ds = None
            when = None

            if isinstance(when_data, dict):
                when = [when_data]
            elif isinstance(when_data, list):
                when = when_data
            else:
                when_ds = None if when_data is None else getDS(when_data)

            disable_lightwood_transform_cache = False
            heavy_transaction_metadata = {}
            if when_ds is None:
                heavy_transaction_metadata['when_data'] = None
            else:
                heavy_transaction_metadata['when_data'] = when_ds
                _, _, disable_lightwood_transform_cache = _get_memory_optimizations(when_ds.df)
            heavy_transaction_metadata['when'] = when
            heavy_transaction_metadata['name'] = self.name

            if backend is not None:
                heavy_transaction_metadata['model_backend'] = backend

            heavy_transaction_metadata['breakpoint'] = self.breakpoint

            light_transaction_metadata = dict(
                name = self.name,
                type = transaction_type,
                use_gpu = use_gpu,
                data_preparation = {},
                force_disable_cache = advanced_args.get('force_disable_cache', disable_lightwood_transform_cache),
                use_database_history = advanced_args.get('use_database_history', False),
                allow_incomplete_history = advanced_args.get('allow_incomplete_history', False),
                quick_predict = advanced_args.get('quick_predict', False)
            )

            self.transaction = PredictTransaction(
                session=self,
                light_transaction_metadata=light_transaction_metadata,
                heavy_transaction_metadata=heavy_transaction_metadata
            )
            self.transaction.run()
            return self.transaction.output_data
