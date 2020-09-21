from pathlib import Path
from collections import defaultdict
from lightwood.constants.lightwood import ColumnDataTypes

import numpy as np
import pandas as pd
import lightwood

from lightwood.constants.lightwood import ColumnDataTypes

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.config import *
from mindsdb_native.libs.helpers.stats_helpers import sample_data


class LightwoodBackend():

    def __init__(self, transaction):
        self.transaction = transaction
        self.predictor = None

    def _create_timeseries_df(self, original_df):
        group_by = self.transaction.lmd['tss']['group_by'] if self.transaction.lmd['tss']['group_by'] is not None else []
        order_by = self.transaction.lmd['tss']['order_by']
        window = self.transaction.lmd['tss']['window']

        secondary_type_dict = {}
        for col in order_by:
            if self.transaction.lmd['stats_v2'][col]['typing']['data_type'] == DATA_TYPES.DATE:
                secondary_type_dict[col] = ColumnDataTypes.DATETIME
            else:
                secondary_type_dict[col] = ColumnDataTypes.NUMERIC

        # Convert order_by columns to numbers
        for _, row in original_df.iterrows():
            for col in order_by:
                if row[col] is None:
                    row[col] = 0.0
                try:
                    row[col] = float(row[col])
                except Exception:
                    try:
                        row[col] = float(row[col].timestamp())
                        secondary_type_dict[col] = ColumnDataTypes.DATETIME
                    except Exception:
                        error_msg = f'Backend Lightwood does not support ordering by the column: {col} !, Faulty value: {row[col]}'
                        self.transaction.log.error(error_msg)
                        raise ValueError(error_msg)

        # TODO: use pandas.DataFrame.groupby, the issue is that it raises
        # an exception when len(group_by) is equal to 0
        # (when no ['tss']['group_by'] is provided)
        # Make groups
        ts_groups = defaultdict(list)
        for _, row in original_df.iterrows():
            ts_groups[tuple(row[group_by])].append(row)

        # Convert each group to pandas.DataFrame
        for group in ts_groups:
            ts_groups[group] = pd.DataFrame.from_records(
                ts_groups[group],
                columns=original_df.columns
            )

        # Sort each group by order_by columns
        for group in ts_groups:
            ts_groups[group].sort_values(by=order_by, inplace=True)

        # Make type `object` so that dataframe cells can be python lists
        for group in ts_groups:
            ts_groups[group] = ts_groups[group].astype(object)

        # Make all order column cells lists
        for group in ts_groups:
            for order_col in order_by:
                for i in range(len(ts_groups[group])):
                    ts_groups[group][order_col].iloc[i] = [
                        ts_groups[group][order_col].iloc[i]
                    ]

        # Add previous rows
        for group in ts_groups:
            for order_col in order_by:
                for i in range(len(ts_groups[group])):
                    previous_indexes = [*range(max(0, i - window), i)]

                    for prev_i in reversed(previous_indexes):
                        ts_groups[group][order_col].iloc[i].append(
                            ts_groups[group][order_col].iloc[prev_i][-1]
                        )

                    # Zeor pad
                    # @TODO: Remove since RNN encoder can do without (???)
                    ts_groups[group].iloc[i][order_col].extend(
                        [0] * (1 + window - len(ts_groups[group].iloc[i][order_col]))
                    )

                    ts_groups[group].iloc[i][order_col].reverse()

        if self.transaction.lmd['tss']['use_previous_target']:
            for target_column in self.transaction.lmd['predict_columns']:
                for k in ts_groups:
                    previous_target_values = list(ts_groups[k][target_column])
                    del previous_target_values[-1]
                    previous_target_values = [None] + previous_target_values
                    ts_groups[k]['previous_' + target_column] = previous_target_values

        combined_df = pd.concat(list(ts_groups.values()))

        if 'make_predictions' in combined_df.columns:
            combined_df = pd.DataFrame(combined_df[combined_df['make_predictions']])
            del combined_df['make_predictions']

        return combined_df, secondary_type_dict

    def _create_lightwood_config(self, secondary_type_dict):
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

            if self.transaction.lmd['tss']['is_timeseries'] and col_name in self.transaction.lmd['tss']['order_by']:
                lightwood_data_type = ColumnDataTypes.TIME_SERIES

            col_config = {
                'name': col_name,
                'type': lightwood_data_type
            }

            if data_subtype == DATA_SUBTYPES.SHORT:
                col_config['encoder_class'] = lightwood.encoders.text.short.ShortTextEncoder


            if col_name in self.transaction.lmd['weight_map']:
                col_config['weights'] = self.transaction.lmd['weight_map'][col_name]

            if col_name in secondary_type_dict:
                col_config['secondary_type'] = secondary_type_dict[col_name]

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
        test_error_rounded = round(test_error, 4)
        for col in accuracy:
            value = accuracy[col]['value']
            if accuracy[col]['function'] == 'r2_score':
                value_rounded = round(value, 3)
                self.transaction.log.debug(f'We\'ve reached training epoch nr {epoch} with an r2 score of {value_rounded} on the testing dataset')
            else:
                value_pct = round(value * 100, 2)
                self.transaction.log.debug(f'We\'ve reached training epoch nr {epoch} with an accuracy of {value_pct}% on the testing dataset')

    def train(self):
        if self.transaction.lmd['use_gpu'] is not None:
            lightwood.config.config.CONFIG.USE_CUDA = self.transaction.lmd['use_gpu']

        secondary_type_dict = {}
        if self.transaction.lmd['tss']['is_timeseries']:
            self.transaction.log.debug('Reshaping data into timeseries format, this may take a while !')
            train_df, secondary_type_dict = self._create_timeseries_df(self.transaction.input_data.train_df)
            test_df, _ = self._create_timeseries_df(self.transaction.input_data.test_df)
            self.transaction.log.debug('Done reshaping data into timeseries format !')
        else:
            if self.transaction.lmd['sample_settings']['sample_for_training']:
                sample_margin_of_error = self.transaction.lmd['sample_settings']['sample_margin_of_error']
                sample_confidence_level = self.transaction.lmd['sample_settings']['sample_confidence_level']
                sample_percentage = self.transaction.lmd['sample_settings']['sample_percentage']
                sample_function = self.transaction.hmd['sample_function']

                train_df = sample_function(
                    self.transaction.input_data.train_df,
                    sample_margin_of_error,
                    sample_confidence_level,
                    sample_percentage
                )

                test_df = sample_function(
                    self.transaction.input_data.test_df,
                    sample_margin_of_error,
                    sample_confidence_level,
                    sample_percentage
                )

                sample_size = len(train_df)
                population_size = len(self.transaction.input_data.train_df)

                self.transaction.log.warning(f'Training on a sample of {round(sample_size * 100 / population_size, 1)}% your data, results can be unexpected.')
            else:
                train_df = self.transaction.input_data.train_df
                test_df = self.transaction.input_data.test_df

        lightwood_config = self._create_lightwood_config(secondary_type_dict)


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

        self.transaction.lmd['lightwood_data']['save_path'] = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.transaction.lmd['name'], 'lightwood_data')
        Path(CONFIG.MINDSDB_STORAGE_PATH).joinpath(self.transaction.lmd['name']).mkdir(mode=0o777, exist_ok=True, parents=True)
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

        if self.transaction.lmd['tss']['is_timeseries']:
            df, _ = self._create_timeseries_df(df)

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
