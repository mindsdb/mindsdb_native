from dateutil.parser import parse as parse_datetime
import datetime
import math

import pandas as pd

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.helpers.text_helpers import clean_float
from lightwood.helpers.text import tokenize_text


def _handle_nan(x):
    if x is not None and math.isnan(x):
        return 0
    else:
        return x


def _try_round(x):
    try:
        return round(x)
    except Exception:
        return None


def _standardize_date(date_str):
    try:
        # will return a datetime object
        date = parse_datetime(date_str)
    except Exception:
        try:
            date = datetime.datetime.utcfromtimestamp(date_str)
        except Exception:
            return None
    return date.strftime('%Y-%m-%d')


def _standardize_datetime(date_str):
    try:
        # will return a datetime object
        dt = parse_datetime(str(date_str))
    except Exception:
        try:
            dt = datetime.datetime.utcfromtimestamp(date_str)
        except Exception:
            return None

    return dt.strftime('%Y-%m-%d %H:%M:%S')


def _tags_to_tuples(tags_str):
    try:
        return tuple([x.strip() for x in tags_str.split(',')])
    except Exception:
        return tuple()


def _lightwood_datetime_processing(dt):
    dt = pd.to_datetime(dt, errors='coerce')
    try:
        return dt.timestamp()
    except Exception:
        return None


def _standardize_timeseries(ts_str):
    """
    erases square brackets, trailing whitespace,
    and commas from the array as string
    """
    try:
        ts_str = ts_str.rstrip(']').lstrip('[')
        ts_str = ts_str.rstrip(' ').lstrip(' ')
        return ts_str.replace(', ', ' ').replace(',', ' ')
    except Exception:
        return ts_str


def _clean_float_or_none(val):
    try:
        return clean_float(val)
    except Exception:
        return None


class DataTransformer(BaseModule):
    def _apply_to_all_data(self, input_data, column, func, transaction_type):
        if transaction_type == TRANSACTION_LEARN:
            input_data.train_df[column] = input_data.train_df[column].apply(func)
            input_data.validation_df[column] = input_data.validation_df[column].apply(func)
            input_data.test_df[column] = input_data.test_df[column].apply(func)

            self.transaction.lmd['stats_v2'][column]['histogram']['x'] = [func(x) for x in self.transaction.lmd['stats_v2'][column]['histogram']['x']]

            if 'percentage_buckets' in self.transaction.lmd['stats_v2'][column] and self.transaction.lmd['stats_v2'][column]['percentage_buckets'] is not None:
                self.transaction.lmd['stats_v2'][column]['percentage_buckets'] = [func(x) for x in self.transaction.lmd['stats_v2'][column]['percentage_buckets']]
        else:
            input_data.data_frame[column] = input_data.data_frame[column].apply(func)

    def run(self, input_data):
        transaction_type = self.transaction.lmd['type']
        for column in self.transaction.lmd['columns']:
            if column in self.transaction.lmd['columns_to_ignore'] or column not in self.transaction.lmd['stats_v2']:
                continue

            data_type = self.transaction.lmd['stats_v2'][column]['typing']['data_type']
            data_subtype = self.transaction.lmd['stats_v2'][column]['typing']['data_subtype']

            if data_type == DATA_TYPES.NUMERIC:
                self._apply_to_all_data(input_data, column, _clean_float_or_none, transaction_type)
                self._apply_to_all_data(input_data, column, _handle_nan, transaction_type)

                if data_subtype == DATA_SUBTYPES.INT:
                    self._apply_to_all_data(input_data, column, _try_round, transaction_type)

            if data_type == DATA_TYPES.DATE:
                if data_subtype == DATA_SUBTYPES.DATE:
                    self._apply_to_all_data(input_data, column, _standardize_date, transaction_type)

                elif data_subtype == DATA_SUBTYPES.TIMESTAMP:
                    self._apply_to_all_data(input_data, column, _standardize_datetime, transaction_type)

            if data_type == DATA_TYPES.CATEGORICAL:
                if data_subtype == DATA_SUBTYPES.TAGS:
                    self._apply_to_all_data(input_data, column, _tags_to_tuples, transaction_type)
                else:
                    self._apply_to_all_data(input_data, column, lambda x: x if x is None else str(x), transaction_type)

            if data_type == DATA_TYPES.TEXT:
                self._apply_to_all_data(input_data, column, lambda x: x if x is None else str(x), transaction_type)

            if data_type == DATA_TYPES.SEQUENTIAL:
                if data_subtype == DATA_SUBTYPES.ARRAY:
                    self._apply_to_all_data(input_data, column, _standardize_timeseries, transaction_type)

            if self.transaction.hmd['model_backend'] == 'lightwood':
                if data_type == DATA_TYPES.DATE:
                    self._apply_to_all_data(input_data, column, _standardize_datetime, transaction_type)
                    self._apply_to_all_data(input_data, column, _lightwood_datetime_processing, transaction_type)
                    self._apply_to_all_data(input_data, column, _handle_nan, transaction_type)

        # Initialize this here, will be overwritten if `equal_accuracy_for_all_output_categories` is specified to be True in order to account for it
        self.transaction.lmd['weight_map'] = self.transaction.lmd['output_categories_importance_dictionary']

        # Un-bias dataset for training
        for column in self.transaction.lmd['predict_columns']:
            if (self.transaction.lmd['stats_v2'][column]['typing']['data_type'] == DATA_TYPES.CATEGORICAL
                and self.transaction.lmd['equal_accuracy_for_all_output_categories'] is True
                and self.transaction.lmd['type'] == TRANSACTION_LEARN):

                occurance_map = {}
                ciclying_map = {}

                for i in range(0,len(self.transaction.lmd['stats_v2'][column]['histogram']['x'])):
                    ciclying_map[self.transaction.lmd['stats_v2'][column]['histogram']['x'][i]] = 0
                    occurance_map[self.transaction.lmd['stats_v2'][column]['histogram']['x'][i]] = self.transaction.lmd['stats_v2'][column]['histogram']['y'][i]

                max_val_occurances = max(occurance_map.values())

                if self.transaction.hmd['model_backend'] in ('lightwood'):
                    lightwood_weight_map = {}
                    for val in occurance_map:
                        lightwood_weight_map[val] = 1/occurance_map[val] #sum(occurance_map.values())

                        if column in self.transaction.lmd['output_categories_importance_dictionary']:
                            if val in self.transaction.lmd['output_categories_importance_dictionary'][column]:
                                lightwood_weight_map[val] = self.transaction.lmd['output_categories_importance_dictionary'][column][val]
                            elif '<default>' in self.transaction.lmd['output_categories_importance_dictionary'][column]:
                                lightwood_weight_map[val] = self.transaction.lmd['output_categories_importance_dictionary'][column]['<default>']

                    self.transaction.lmd['weight_map'][column] = lightwood_weight_map

                #print(self.transaction.lmd['weight_map'])
                column_is_weighted_in_train = column in self.transaction.lmd['weight_map']

                if not column_is_weighted_in_train:
                    dfs = ['input_data.train_df','input_data.test_df','input_data.validation_df']

                    total_len = (len(input_data.train_df) + len(input_data.test_df) + len(input_data.validation_df))
                    # Since pandas doesn't support append in-place we'll just do some eval-based hacks

                    for dfn in dfs:
                        max_val_occurances_in_set = int(round(max_val_occurances * len(eval(dfn))/total_len))
                        for val in occurance_map:
                            valid_rows = eval(dfn)[eval(dfn)[column] == val]
                            if len(valid_rows) == 0:
                                continue

                            appended_times = 0
                            while max_val_occurances_in_set > len(valid_rows) * (2 + appended_times):
                                exec(f'{dfn} = {dfn}.append(valid_rows)')
                                appended_times += 1

                            if int(max_val_occurances_in_set - len(valid_rows) * (1 + appended_times)) > 0:
                                exec(f'{dfn} = {dfn}.append(valid_rows[0:int(max_val_occurances_in_set - len(valid_rows) * (1 + appended_times))])')
