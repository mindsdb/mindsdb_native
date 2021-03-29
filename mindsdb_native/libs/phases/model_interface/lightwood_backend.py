import copy
import traceback
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time

import numpy as np
import pandas as pd
import lightwood
from lightwood.constants.lightwood import ColumnDataTypes

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.config import *
from mindsdb_native.libs.helpers.general_helpers import evaluate_accuracy
from mindsdb_native.libs.helpers.mp_helpers import get_nr_procs


def _make_pred(row):
    return not hasattr(row, 'make_predictions') or row.make_predictions


def _ts_to_obj(df, historical_columns):
    for hist_col in historical_columns:
        df.loc[:, hist_col] = df[hist_col].astype(object)
    return df


def _ts_order_col_to_cell_lists(df, historical_columns):
    for order_col in historical_columns:
        for ii in range(len(df)):
            label = df.index.values[ii]
            df.at[label, order_col] = [df.at[label, order_col]]
    return df


def _ts_add_previous_rows(df, historical_columns, window):
    for order_col in historical_columns:
        for i in range(len(df)):
            previous_indexes = [*range(max(0, i - window), i)]

            for prev_i in reversed(previous_indexes):
                df.iloc[i][order_col].append(
                    df.iloc[prev_i][order_col][-1]
                )

            # Zero pad
            # @TODO: Remove since RNN encoder can do without (???)
            df.iloc[i][order_col].extend(
                [0] * (1 + window - len(df.iloc[i][order_col]))
            )
            df.iloc[i][order_col].reverse()
    return df


def _ts_add_previous_target(df, predict_columns, nr_predictions, window):
    for target_column in predict_columns:
        previous_target_values = list(df[target_column])
        del previous_target_values[-1]
        previous_target_values = [None] + previous_target_values

        previous_target_values_arr = []
        for i in range(len(previous_target_values)):
            prev_vals = previous_target_values[max(i - window, 0):i + 1]
            arr = [None] * (window - len(prev_vals) + 1)
            arr.extend(prev_vals)
            previous_target_values_arr.append(arr)

        df[f'__mdb_ts_previous_{target_column}'] = previous_target_values_arr
        for timestep_index in range(1, nr_predictions):
            next_target_value_arr = list(df[target_column])
            for del_index in range(0, min(timestep_index, len(next_target_value_arr))):
                del next_target_value_arr[del_index]
                next_target_value_arr.append(0)
            # @TODO: Maybe ignore the rows with `None` next targets for training
            df[f'{target_column}_timestep_{timestep_index}'] = next_target_value_arr
    return df


class LightwoodBackend:
    def __init__(self, transaction):
        self.transaction = transaction
        self.predictor = None
        self.nr_predictions = self.transaction.lmd['tss']['nr_predictions']
        # Ideally this will go away soon but still a necessity for now
        self.nn_mixer_only = False

    def _ts_reshape(self, original_df):
        original_df = copy.deepcopy(original_df)
        gb_arr = self.transaction.lmd['tss']['group_by'] if self.transaction.lmd['tss']['group_by'] is not None else []
        ob_arr = self.transaction.lmd['tss']['order_by']
        window = self.transaction.lmd['tss']['window']

        original_index_list = []
        idx = 0
        for row in original_df.itertuples():
            if _make_pred(row):
                original_index_list.append(idx)
                idx += 1
            else:
                original_index_list.append(None)

        original_df['original_index'] = original_index_list

        secondary_type_dict = {}
        for col in ob_arr:
            if self.transaction.lmd['stats_v2'][col]['typing']['data_type'] == DATA_TYPES.DATE:
                secondary_type_dict[col] = ColumnDataTypes.DATETIME
            else:
                secondary_type_dict[col] = ColumnDataTypes.NUMERIC

        # Convert order_by columns to numbers (note, rows are references to mutable rows in `original_df`)
        for _, row in original_df.iterrows():
            for col in ob_arr:
                # @TODO: Remove if the TS encoder can handle `None`
                if row[col] is None or pd.isna(row[col]):
                    row[col] = 0.0

                try:
                    row[col] = row[col].timestamp()
                except Exception:
                    pass

                try:
                    row[col] = float(row[col])
                except Exception:
                    raise ValueError(f'Failed to order based on column: "{col}" due to faulty value: {row[col]}')

        if len(gb_arr) > 0:
            df_arr = []
            for _, df in original_df.groupby(gb_arr):
                df_arr.append(df.sort_values(by=ob_arr))
        else:
            df_arr = [original_df]

        if len(original_df) > 500:
            nr_procs = get_nr_procs(self.transaction.lmd.get('max_processes', None),
                                    self.transaction.lmd.get('max_per_proc_usage', None),
                                    original_df)
            self.transaction.log.info(f'Using {nr_procs} processes to reshape.')
            pool = mp.Pool(processes=nr_procs)
            # Make type `object` so that dataframe cells can be python lists
            df_arr = pool.map(partial(_ts_to_obj, historical_columns=ob_arr + self.transaction.lmd['tss']['historical_columns']), df_arr)
            df_arr = pool.map(partial(_ts_order_col_to_cell_lists, historical_columns=ob_arr + self.transaction.lmd['tss']['historical_columns']), df_arr)
            df_arr = pool.map(partial(_ts_add_previous_rows, historical_columns=ob_arr + self.transaction.lmd['tss']['historical_columns'], window=window), df_arr)
            if self.transaction.lmd['tss']['use_previous_target']:
                df_arr = pool.map(partial(_ts_add_previous_target, predict_columns=self.transaction.lmd['predict_columns'], nr_predictions=self.nr_predictions, window=window), df_arr)
            pool.close()
            pool.join()
        else:
            for i in range(len(df_arr)):
                df_arr[i] = _ts_to_obj(df_arr[i], historical_columns=ob_arr + self.transaction.lmd['tss']['historical_columns'])
                df_arr[i] = _ts_order_col_to_cell_lists(df_arr[i], historical_columns=ob_arr + self.transaction.lmd['tss']['historical_columns'])
                df_arr[i] = _ts_add_previous_rows(df_arr[i], historical_columns=ob_arr + self.transaction.lmd['tss']['historical_columns'], window=window)
                if self.transaction.lmd['tss']['use_previous_target']:
                    df_arr[i] = _ts_add_previous_target(df_arr[i], predict_columns=self.transaction.lmd['predict_columns'], nr_predictions=self.nr_predictions, window=window)

        combined_df = pd.concat(df_arr)

        if 'make_predictions' in combined_df.columns:
            combined_df = pd.DataFrame(combined_df[combined_df['make_predictions'].astype(bool) == True])
            del combined_df['make_predictions']

        if len(combined_df) == 0:
            raise Exception(f'Not enough historical context to make a timeseries prediction. Please provide a number of rows greater or equal to the window size. If you can\'t get enough rows, consider lowering your window size. If you want to force timeseries predictions lacking historical context please set the `allow_incomplete_history` advanced argument to `True`, but this might lead to subpar predictions.')

        df_gb_map = None
        if len(df_arr) > 1 and (self.transaction.lmd['quick_learn'] or self.transaction.lmd['quick_predict']):
            df_gb_list = list(combined_df.groupby(self.transaction.lmd['split_models_on']))
            df_gb_map = {}
            for gb, df in df_gb_list:
                df_gb_map['_' + '_'.join(gb)] = df

        timeseries_row_mapping = {}
        idx = 0

        if df_gb_map is None:
            for _, row in combined_df.iterrows():
                timeseries_row_mapping[idx] = int(row['original_index']) if row['original_index'] is not None and not np.isnan(row['original_index']) else None
                idx += 1
        else:
            for gb in df_gb_map:
                for _, row in df_gb_map[gb].iterrows():
                    timeseries_row_mapping[idx] = int(row['original_index']) if row['original_index'] is not None and not np.isnan(row['original_index']) else None
                    idx += 1

        del combined_df['original_index']


        return combined_df, secondary_type_dict, timeseries_row_mapping, df_gb_map

    def _create_lightwood_config(self, secondary_type_dict):
        config = {}

        config['input_features'] = []
        config['output_features'] = []

        for col_name in self.transaction.lmd['columns']:
            if col_name in self.transaction.lmd['columns_to_ignore'] or col_name not in self.transaction.lmd['stats_v2']:
                continue

            col_stats = self.transaction.lmd['stats_v2'][col_name]
            data_subtype = col_stats['typing']['data_subtype']
            data_type = col_stats['typing']['data_type']

            other_keys = {'encoder_attrs': {}}
            if data_type == DATA_TYPES.NUMERIC:
                lightwood_data_type = ColumnDataTypes.NUMERIC
                if col_name in self.transaction.lmd['predict_columns'] and col_stats.get('positive_domain', False):
                    other_keys['encoder_attrs']['positive_domain'] = True

            elif data_type == DATA_TYPES.CATEGORICAL:
                predict_proba = self.transaction.lmd['output_class_distribution']
                if col_name in self.transaction.lmd['predict_columns'] and predict_proba:
                    other_keys['encoder_attrs']['predict_proba'] = predict_proba

                if data_subtype == DATA_SUBTYPES.TAGS:
                    lightwood_data_type = ColumnDataTypes.MULTIPLE_CATEGORICAL
                    if predict_proba:
                        self.transaction.log.warning(f'Class distribution not supported for tags prediction\n')
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
                err = f'The lightwood model backend is unable to handle data of type {data_type} and subtype {data_subtype} !'
                self.transaction.log.error(err)
                raise Exception(err)

            if self.transaction.lmd['tss']['is_timeseries'] and col_name in self.transaction.lmd['tss']['order_by']:
                lightwood_data_type = ColumnDataTypes.TIME_SERIES

            grouped_by = self.transaction.lmd['tss'].get('group_by', [])
            col_config = {
                'name': col_name,
                'type': lightwood_data_type,
                'grouped_by': col_name in grouped_by if grouped_by else False
            }

            if data_subtype == DATA_SUBTYPES.SHORT:
                col_config['encoder_class'] = lightwood.encoders.text.short.ShortTextEncoder

            if col_name in self.transaction.lmd['weight_map']:
                col_config['weights'] = self.transaction.lmd['weight_map'][col_name]

            if col_name in secondary_type_dict:
                col_config['secondary_type'] = secondary_type_dict[col_name]

            col_config.update(other_keys)

            if col_name in self.transaction.lmd['predict_columns']:
                if not ((data_type == DATA_TYPES.CATEGORICAL and data_subtype != DATA_SUBTYPES.TAGS) or data_type == DATA_TYPES.NUMERIC):
                    self.nn_mixer_only = True

                if self.transaction.lmd['tss']['is_timeseries']:
                    col_config['additional_info'] = {
                        'nr_predictions': self.transaction.lmd['tss']['nr_predictions'],
                        'time_series_target': True,
                    }
                config['output_features'].append(col_config)

                if self.transaction.lmd['tss']['is_timeseries'] and self.transaction.lmd['tss']['use_previous_target']:
                    p_col_config = copy.deepcopy(col_config)
                    p_col_config['name'] = f"__mdb_ts_previous_{p_col_config['name']}"
                    p_col_config['original_type'] = col_config['type']
                    p_col_config['type'] = ColumnDataTypes.TIME_SERIES

                    if 'secondary_type' in col_config:
                        p_col_config['secondary_type'] = col_config['secondary_type']

                    config['input_features'].append(p_col_config)

                if self.nr_predictions > 1:
                    self.transaction.lmd['stats_v2'][col_name]['typing']['data_subtype'] = DATA_SUBTYPES.ARRAY
                    self.transaction.lmd['stats_v2'][col_name]['typing']['data_type'] = DATA_TYPES.SEQUENTIAL
                    for timestep_index in range(1,self.nr_predictions):
                        additional_target_config = copy.deepcopy(col_config)
                        additional_target_config['name'] = f'{col_name}_timestep_{timestep_index}'
                        config['output_features'].append(additional_target_config)
            else:
                if col_name in self.transaction.lmd['tss']['historical_columns']:
                    if 'secondary_type' in col_config:
                        col_config['secondary_type'] = col_config['secondary_type']
                    col_config['original_type'] = col_config['type']
                    col_config['type'] = ColumnDataTypes.TIME_SERIES

                config['input_features'].append(col_config)

        config['data_source'] = {}
        config['data_source']['cache_transformed_data'] = not self.transaction.lmd['force_disable_cache']

        config['mixer'] = {
            'class': lightwood.mixers.NnMixer,
            'kwargs': {
                'selfaware': self.transaction.lmd['use_selfaware_model']
            }
        }

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

        if self.transaction.lmd['quick_learn']:
            self.transaction.input_data.train_df = pd.concat([copy.deepcopy(self.transaction.input_data.train_df),copy.deepcopy(self.transaction.input_data.test_df)]).reset_index()
            self.transaction.input_data.test_df = copy.deepcopy(self.transaction.input_data.validation_df).reset_index()

        secondary_type_dict = {}
        if self.transaction.lmd['tss']['is_timeseries']:
            self.transaction.log.debug('Reshaping data into timeseries format, this may take a while !')

            train_df, secondary_type_dict, _, train_df_gb_map = self._ts_reshape(self.transaction.input_data.train_df)
            test_df, _, _, test_df_gb_map = self._ts_reshape(self.transaction.input_data.test_df)

            self.transaction.log.debug('Done reshaping data into timeseries format !')
        else:
            # @TODO: Make sampling work for timeseries
            # @TODO: Alternative, doing train sampling here is really dumb, move to the data extractor
            train_df_gb_map = None
            train_df = self.transaction.input_data.train_df
            test_df = self.transaction.input_data.test_df

        if train_df_gb_map is None:
            train_df_gb_map = {'': train_df}
            test_df_gb_map = {'': test_df}

        self.transaction.input_data.cached_train_df = train_df
        self.transaction.input_data.cached_test_df = test_df

        for gb_val in train_df_gb_map:
            train_df = train_df_gb_map[gb_val]
            test_df = test_df_gb_map[gb_val]

            lightwood_config = self._create_lightwood_config(secondary_type_dict)

            lightwood_train_ds = lightwood.api.data_source.DataSource(
                train_df,
                config=lightwood_config
            )
            lightwood_test_ds = lightwood_train_ds.make_child(test_df)

            Path(CONFIG.MINDSDB_STORAGE_PATH).joinpath(self.transaction.lmd['name']).mkdir(mode=0o777, exist_ok=True, parents=True)

            logging.getLogger().setLevel(logging.DEBUG)

            reasonable_training_time = train_df.shape[0] * train_df.shape[1] / 20

            predictors_and_accuracies = []

            if self.transaction.lmd.get('use_mixers', None) is not None:
                mixer_classes = self.transaction.lmd['use_mixers']
            elif self.nn_mixer_only:
                mixer_classes = [lightwood.mixers.NnMixer]
            else:
                mixer_classes = [lightwood.mixers.LightGBMMixer, lightwood.mixers.NnMixer]

            final_mixer_classes = []
            for mixer_class in mixer_classes:
                if isinstance(mixer_class, str):
                    for mx_cls in lightwood.mixers.BaseMixer.__subclasses__():
                        if mx_cls.__name__ == mixer_class:
                            mixer_class = mx_cls
                            break
                    else:
                        raise ValueError(f'Mixer "{mixer_class}" doesn\'t exist')
                if mixer_class is not None:
                    final_mixer_classes.append(mixer_class)

            if len(final_mixer_classes) == 0:
                raise Exception(f'No valid mixers')

            # @TODO Might be better to turn this into a function
            stop_training_after = self.transaction.lmd['stop_training_in_x_seconds']
            if stop_training_after is None:
                # Stop training after 12 hours unless the user doesn't want us to
                stop_training_after = 3600 * 12

            took_thus_far = int(time.time() - self.transaction.lmd['learn_started_at'])
            remaining_time = int(max(0, stop_training_after - took_thus_far))
            if took_thus_far*2 > stop_training_after:
                new_remaining_time = stop_training_after
                self.transaction.log.warning(f'You asked your predictor to stop training too quickly, the data preparation phase alone took {took_thus_far}. We\'d be left with only {remaining_time} to train the underlying machine learning models and analyze them. Will instead try to spend {new_remaining_time} seconds doing so')
                remaining_time = new_remaining_time

            # We don't know how long the model analysis will last, so let's allocate 1/4th of the remaining time for training to it
            data_analysis_takes = 1/4

            training_time_per_mixer = (remaining_time*(1-data_analysis_takes))/len(mixer_classes)
            # @TODO Might be better to turn this into a function

            for mixer_class in final_mixer_classes:
                lightwood_config['mixer']['kwargs'] = {}
                lightwood_config['mixer']['class'] = mixer_class

                if lightwood_config['mixer']['class'] == lightwood.mixers.NnMixer:
                    # Evaluate less often for larger datasets and vice-versa
                    eval_every_x_epochs = int(round(1 * pow(10, 6) * (1 / len(train_df))))
                    # Within some limits
                    if eval_every_x_epochs > 200:
                        eval_every_x_epochs = 200
                    if eval_every_x_epochs < 3:
                        eval_every_x_epochs = 3

                    lightwood_config['mixer']['kwargs']['callback_on_iter'] = self.callback_on_iter
                    lightwood_config['mixer']['kwargs']['eval_every_x_epochs'] = eval_every_x_epochs / len(final_mixer_classes)

                lightwood_config['mixer']['kwargs']['stop_training_after_seconds'] = training_time_per_mixer

                self.predictor = lightwood.Predictor(lightwood_config.copy())

                try:
                    self.predictor.learn(
                        from_data=lightwood_train_ds,
                        test_data=lightwood_test_ds
                    )
                except Exception:
                    if self.transaction.lmd['debug']:
                        raise
                    else:
                        self.transaction.log.error(traceback.format_exc())
                        self.transaction.log.error('Exception while running {}'.format(mixer_class.__name__))
                        continue

                self.transaction.log.info('[{}] Training accuracy of: {}'.format(
                    mixer_class.__name__,
                    self.predictor.train_accuracy
                ))

                validation_predictions = self.predict('validate')

                validation_df = self.transaction.input_data.validation_df

                if self.transaction.lmd['tss']['is_timeseries']:
                    validation_df = self.transaction.input_data.validation_df[self.transaction.input_data.validation_df['make_predictions'] == True]

                validation_accuracy = evaluate_accuracy(
                    validation_predictions,
                    validation_df,
                    self.transaction.lmd['stats_v2'],
                    self.transaction.lmd['predict_columns'],
                    backend=self
                )

                predictors_and_accuracies.append((
                    self.predictor,
                    validation_accuracy
                ))

            if len(predictors_and_accuracies) == 0:
                raise Exception('All models had an error while training')

            best_predictor, best_accuracy = max(predictors_and_accuracies, key=lambda x: x[1])

            # Find predictor with NnMixer
            for predictor, accuracy in predictors_and_accuracies:
                if lightwood.mixers.NnMixer is not None and isinstance(predictor._mixer, lightwood.mixers.NnMixer):
                    nn_mixer_predictor, nn_mixer_predictor_accuracy = predictor, accuracy
                    break
            else:
                nn_mixer_predictor, nn_mixer_predictor_accuracy = None, None

            self.predictor = best_predictor

            # If difference between accuracies of best predictor and NnMixer predictor
            # is small, then use NnMixer predictor
            if nn_mixer_predictor is not None:
                SMALL_ACCURACY_DIFFERENCE = 0.01
                if (best_accuracy - nn_mixer_predictor_accuracy) < SMALL_ACCURACY_DIFFERENCE:
                    self.predictor = nn_mixer_predictor

            save_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.transaction.lmd['name'], 'lightwood_data' + gb_val)
            self.predictor.save(path_to=save_path)

    def predict(self, mode='predict', ignore_columns=None, all_mixers=False):
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

        df_gb_map = None
        if self.transaction.lmd['tss']['is_timeseries']:
            df, _, timeseries_row_mapping, df_gb_map = self._ts_reshape(df)

        if df_gb_map is None:
            df_gb_map = {'': df}

        formated_predictions_arr = []
        for gb_val in df_gb_map:
            formated_predictions = {}
            df = df_gb_map[gb_val]

            if self.predictor is None:
                predictor_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.transaction.lmd['name'], 'lightwood_data' + gb_val)
                self.predictor = lightwood.Predictor(load_from_path=predictor_path)

            # not the most efficient but least prone to bug and should be fast enough
            if len(ignore_columns) > 0:
                run_df = df.copy(deep=True)
                for col_name in ignore_columns:
                    run_df[col_name] = [None] * len(run_df[col_name])
            else:
                run_df = df

            predictions = self.predictor.predict(when_data=run_df)

            # cache run_df to avoid duplicate reshaping in analysis phase
            if mode == 'validate':
                self.transaction.input_data.cached_val_df = run_df
            elif mode == 'predict':
                self.transaction.input_data.cached_pred_df = run_df

            if self.transaction.lmd['quick_predict']:
                for k in predictions:
                    formated_predictions[k] = predictions[k]['predictions']
                    if self.transaction.lmd['output_class_distribution'] and predictions[k].get('class_distribution', False):
                        formated_predictions[f'{k}_class_distribution'] = predictions[k]['class_distribution']
                        self.transaction.lmd['stats_v2'][k]['lightwood_class_map'] = predictions[k]['class_labels']
                    formated_predictions_arr.append(formated_predictions)
                continue

            for k in predictions:
                if '_timestep_' in k:
                    continue

                formated_predictions[k] = predictions[k]['predictions']
                if self.transaction.lmd['output_class_distribution']:
                    try:
                        formated_predictions[f'{k}_class_distribution'] = predictions[k]['class_distribution']
                        self.transaction.lmd['stats_v2'][k]['lightwood_class_map'] = predictions[k]['class_labels']
                    except KeyError:
                        pass

                if self.nr_predictions > 1:
                    formated_predictions[k] = [[x] for x in formated_predictions[k]]
                    for timestep_index in range(1,self.nr_predictions):
                        for i in range(len(formated_predictions[k])):
                            formated_predictions[k][i].append(predictions[f'{k}_timestep_{timestep_index}']['predictions'][i])

                model_confidence_dict = {}
                for confidence_name in ['selfaware_confidences','loss_confidences']:
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

                if 'selfaware_confidences' in predictions[k]:
                    formated_predictions[f'{k}_selfaware_scores'] = [c[0] for c in model_confidence_dict[k]]

                for k in model_confidence_dict:
                    model_confidence_dict[k] = [np.mean(x) for x in model_confidence_dict[k]]

                for k in model_confidence_dict:
                    formated_predictions[f'{k}_model_confidence'] = model_confidence_dict[k]
            formated_predictions_arr.append(formated_predictions)

        formated_predictions = {}
        for k in formated_predictions_arr[0]:
            formated_predictions[k] = []
            for ele in formated_predictions_arr:
                formated_predictions[k].extend(ele[k])

        if self.transaction.lmd['tss']['is_timeseries']:
            for k in list(formated_predictions.keys()):
                ordered_values = [None] * len(formated_predictions[k])
                for i, value in enumerate(formated_predictions[k]):
                    if timeseries_row_mapping[i] is not None:
                        ordered_values[timeseries_row_mapping[i]] = value
                formated_predictions[k] = ordered_values

        return formated_predictions
