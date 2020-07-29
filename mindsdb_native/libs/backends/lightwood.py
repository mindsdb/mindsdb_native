from lightwood.constants.lightwood import ColumnDataTypes

import numpy as np
import pandas as pd
import lightwood

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.config import *
from mindsdb_native.libs.helpers.stats_helpers import sample_data


class LightwoodBackend():

    def __init__(self, transaction):
        self.transaction = transaction
        self.predictor = None

    def _get_group_by_key(self, group_by, row):
        gb_lookup_key = '!!@@!!'
        for column in group_by:
            gb_lookup_key += f'{column}_{row[column]}_!!@@!!'
        return gb_lookup_key

    def _create_timeseries_df(self, original_df):
        group_by = self.transaction.lmd['model_group_by']
        order_by = [x[0] for x in self.transaction.lmd['model_order_by']]
        nr_samples = self.transaction.lmd['window_size']

        group_by_ts_map = {}

        for _, row in original_df.iterrows():
            gb_lookup_key = self._get_group_by_key(group_by, row)
            if gb_lookup_key not in group_by_ts_map:
                group_by_ts_map[gb_lookup_key] = []

            for col in order_by:
                if row[col] is None:
                    row[col] = 0.0
                try:
                    row[col] = float(row[col])
                except Exception:
                    try:
                        row[col] = float(row[col].timestamp())
                    except Exception:
                        error_msg = f'Backend Lightwood does not support ordering by the column: {col} !, Faulty value: {row[col]}'
                        self.transaction.log.error(error_msg)
                        raise ValueError(error_msg)

            group_by_ts_map[gb_lookup_key].append(row)

        for k in group_by_ts_map:
            group_by_ts_map[k] = pd.DataFrame.from_records(group_by_ts_map[k], columns=original_df.columns)
            group_by_ts_map[k] = group_by_ts_map[k].sort_values(by=order_by)

            for order_col in order_by:
                for i in range(0,len(group_by_ts_map[k])):
                    group_by_ts_map[k][order_col] = group_by_ts_map[k][order_col].astype(object)


                    numerical_value = float(group_by_ts_map[k][order_col].iloc[i])
                    arr_val = [str(numerical_value)]

                    group_by_ts_map[k][order_col].iat[i] = arr_val

                    previous_indexes = list(range(i - nr_samples, i))
                    previous_indexes = [x for x in previous_indexes if x >= 0]
                    previous_indexes.reverse()

                    for prev_i in previous_indexes:
                        group_by_ts_map[k].iloc[i][order_col].append(group_by_ts_map[k][order_col].iloc[prev_i].split(' ')[-1])

                    while len(group_by_ts_map[k].iloc[i][order_col]) <= nr_samples:
                        group_by_ts_map[k].iloc[i][order_col].append('0')

                    group_by_ts_map[k].iloc[i][order_col].reverse()
                    group_by_ts_map[k][order_col].iat[i] = ' '.join(group_by_ts_map[k].iloc[i][order_col])

        combined_df = pd.concat(list(group_by_ts_map.values()))
        return combined_df

    def _create_lightwood_config(self):
        config = {}

        config['input_features'] = []
        config['output_features'] = []

        for col_name in self.transaction.input_data.columns:
            if col_name in self.transaction.lmd['columns_to_ignore'] or col_name not in self.transaction.lmd['stats_v2']:
                continue

            col_stats = self.transaction.lmd['stats_v2'][col_name]
            data_subtype = col_stats['typing']['data_subtype']
            data_type = col_stats['typing']['data_type']

            other_keys = {'encoder_attrs': {}}
            if data_type == DATA_TYPES.NUMERIC:
                lightwood_data_type = ColumnDataTypes.NUMERIC

            elif data_type == DATA_TYPES.CATEGORICAL:
                if data_subtype == DATA_SUBTYPES.TAGS:
                    lightwood_data_type = ColumnDataTypes.MULTIPLE_CATEGORICAL
                else:
                    lightwood_data_type = ColumnDataTypes.CATEGORICAL

            elif data_subtype in (DATA_SUBTYPES.TIMESTAMP, DATA_SUBTYPES.DATE):
                lightwood_data_type = ColumnDataTypes.DATETIME

            elif data_subtype == DATA_SUBTYPES.IMAGE:
                lightwood_data_type = ColumnDataTypes.IMAGE
                other_keys['encoder_attrs']['aim'] = 'balance'

            elif data_subtype == DATA_SUBTYPES.AUDIO:
                lightwood_data_type = ColumnDataTypes.AUDIO

            elif data_subtype == DATA_SUBTYPES.RICH:
                lightwood_data_type = ColumnDataTypes.TEXT

            elif data_subtype == DATA_SUBTYPES.SHORT:
                lightwood_data_type = ColumnDataTypes.SHORT_TEXT

            elif data_subtype == DATA_SUBTYPES.ARRAY:
                lightwood_data_type = ColumnDataTypes.TIME_SERIES

            else:
                self.transaction.log.error(f'The lightwood model backend is unable to handle data of type {data_type} and subtype {data_subtype} !')
                raise Exception('Failed to build data definition for Lightwood model backend')

            if col_name in [x[0] for x in self.transaction.lmd['model_order_by']]:
                lightwood_data_type = ColumnDataTypes.TIME_SERIES

            col_config = {
                'name': col_name,
                'type': lightwood_data_type
            }

            if data_subtype == DATA_SUBTYPES.SHORT:
                col_config['encoder_class'] = lightwood.encoders.text.short.ShortTextEncoder


            if col_name in self.transaction.lmd['weight_map']:
                col_config['weights'] = self.transaction.lmd['weight_map'][col_name]

            col_config.update(other_keys)

            if col_name in self.transaction.lmd['predict_columns']:
                config['output_features'].append(col_config)
            else:
                config['input_features'].append(col_config)


        config['data_source'] = {}
        config['data_source']['cache_transformed_data'] = not self.transaction.lmd['force_disable_cache']

        config['mixer'] = {}
        config['mixer']['selfaware'] = self.transaction.lmd['use_selfaware_model']

        return config

    def callback_on_iter(self, epoch, mix_error, test_error, delta_mean, accuracy):
        test_error_rounded = round(test_error,4)
        for col in accuracy:
            value = accuracy[col]['value']
            if accuracy[col]['function'] == 'r2_score':
                value_rounded = round(value,3)
                self.transaction.log.debug(f'We\'ve reached training epoch nr {epoch} with an r2 score of {value_rounded} on the testing dataset')
            else:
                value_pct = round(value * 100,2)
                self.transaction.log.debug(f'We\'ve reached training epoch nr {epoch} with an accuracy of {value_pct}% on the testing dataset')

    def train(self):
        if self.transaction.lmd['use_gpu'] is not None:
            lightwood.config.config.CONFIG.USE_CUDA = self.transaction.lmd['use_gpu']

        if self.transaction.lmd['model_order_by'] is not None and len(self.transaction.lmd['model_order_by']) > 0:
            self.transaction.log.debug('Reshaping data into timeseries format, this may take a while !')
            train_df = self._create_timeseries_df(self.transaction.input_data.train_df)
            test_df = self._create_timeseries_df(self.transaction.input_data.test_df)
            self.transaction.log.debug('Done reshaping data into timeseries format !')
        else:
            if self.transaction.lmd['sample_settings']['sample_for_training']:
                sample_margin_of_error = self.transaction.lmd['sample_settings']['sample_margin_of_error']
                sample_confidence_level = self.transaction.lmd['sample_settings']['sample_confidence_level']
                sample_percentage = self.transaction.lmd['sample_settings']['sample_percentage']
                sample_function = self.transaction.hmd['sample_function']

                train_df = sample_function(self.transaction.input_data.train_df,
                                       sample_margin_of_error,
                                       sample_confidence_level,
                                       sample_percentage)

                test_df = sample_function(self.transaction.input_data.test_df,
                                       sample_margin_of_error,
                                       sample_confidence_level,
                                       sample_percentage)

                sample_size = len(train_df)
                population_size = len(self.transaction.input_data.train_df)

                self.transaction.log.warning(f'Training on a sample of {round(sample_size * 100 / population_size, 1)}% your data, results can be unexpected.')
            else:
                train_df = self.transaction.input_data.train_df
                test_df = self.transaction.input_data.test_df

        lightwood_config = self._create_lightwood_config()


        self.predictor = lightwood.Predictor(lightwood_config)

        # Evaluate less often for larger datasets and vice-versa
        eval_every_x_epochs = int(round(1 * pow(10,6) * (1/len(train_df))))

        # Within some limits
        if eval_every_x_epochs > 200:
            eval_every_x_epochs = 200
        if eval_every_x_epochs < 3:
            eval_every_x_epochs = 3

        logging.getLogger().setLevel(logging.DEBUG)
        if self.transaction.lmd['stop_training_in_x_seconds'] is None:
            self.predictor.learn(from_data=train_df, test_data=test_df, callback_on_iter=self.callback_on_iter, eval_every_x_epochs=eval_every_x_epochs)
        else:
            self.predictor.learn(from_data=train_df, test_data=test_df, stop_training_after_seconds=self.transaction.lmd['stop_training_in_x_seconds'], callback_on_iter=self.callback_on_iter, eval_every_x_epochs=eval_every_x_epochs)

        self.transaction.log.info('Training accuracy of: {}'.format(self.predictor.train_accuracy))

        self.transaction.lmd['lightwood_data']['save_path'] = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.transaction.lmd['name'] + '_lightwood_data')
        self.predictor.save(path_to=self.transaction.lmd['lightwood_data']['save_path'])

    def predict(self, mode='predict', ignore_columns=None):
        if ignore_columns is None:
            ignore_columns = []
        if self.transaction.lmd['use_gpu'] is not None:
            lightwood.config.config.CONFIG.USE_CUDA = self.transaction.lmd['use_gpu']

        if mode == 'predict':
            df = self.transaction.input_data.data_frame
        elif mode == 'validate':
            df = self.transaction.input_data.validation_df
        elif mode == 'test':
            df = self.transaction.input_data.test_df
        elif mode == 'predict_on_train_data':
            df = self.transaction.input_data.train_df
        else:
            raise Exception(f'Unknown mode specified: "{mode}"')

        if self.transaction.lmd['model_order_by'] is not None and len(self.transaction.lmd['model_order_by']) > 0:
            df = self._create_timeseries_df(df)

        if self.predictor is None:
            self.predictor = lightwood.Predictor(load_from_path=self.transaction.lmd['lightwood_data']['save_path'])

        # not the most efficient but least prone to bug and should be fast enough
        if len(ignore_columns) > 0:
            run_df = df.copy(deep=True)
            for col_name in ignore_columns:
                run_df[col_name] = [None] * len(run_df[col_name])
        else:
            run_df = df

        predictions = self.predictor.predict(when_data=run_df)

        formated_predictions = {}
        for k in predictions:
            formated_predictions[k] = predictions[k]['predictions']

            model_confidence_dict = {}
            for confidence_name in ['selfaware_confidences','loss_confidences', 'quantile_confidences']:

                if confidence_name in predictions[k]:
                    if k not in model_confidence_dict:
                        model_confidence_dict[k] = []

                    for i in range(len(predictions[k][confidence_name])):
                        if len(model_confidence_dict[k]) <= i:
                            model_confidence_dict[k].append([])
                        conf = predictions[k][confidence_name][i]
                        # @TODO We should make sure lightwood never returns confidences above or bellow 0 and 1
                        if conf < 0:
                            conf = 0
                        if conf > 1:
                            conf = 1
                        model_confidence_dict[k][i].append(conf)

            for k in model_confidence_dict:
                model_confidence_dict[k] = [np.mean(x) for x in model_confidence_dict[k]]

            for k  in model_confidence_dict:
                formated_predictions[f'{k}_model_confidence'] = model_confidence_dict[k]

            if 'confidence_range' in predictions[k]:
                formated_predictions[f'{k}_confidence_range'] = predictions[k]['confidence_range']

        return formated_predictions
