from dateutil.parser import parse as parse_datetime
import datetime
import math

import pandas as pd

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.helpers.text_helpers import clean_float


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


def _lightwood_datetime_processing(dt):
    dt = pd.to_datetime(dt, errors='coerce')
    try:
        return dt.timestamp()
    except Exception:
        return None


def _clean_float_or_none(val):
    try:
        return clean_float(val)
    except Exception:
        return None


class DataTransformer(BaseModule):
    def _apply_to_all_data(self, input_data, column, func):
        self.transaction.lmd['stats_v2'][column]['histogram']['x'] = [func(x) for x in self.transaction.lmd['stats_v2'][column]['histogram']['x']]

        if 'percentage_buckets' in self.transaction.lmd['stats_v2'][column] and self.transaction.lmd['stats_v2'][column]['percentage_buckets'] is not None:
            self.transaction.lmd['stats_v2'][column]['percentage_buckets'] = [func(x) for x in self.transaction.lmd['stats_v2'][column]['percentage_buckets']]
        input_data.data_frame[column] = input_data.data_frame[column].apply(func)

    def run(self, input_data):
        # Drop foreign keys
        if self.transaction.lmd['handle_foreign_keys']:
            cols_to_drop = []
            for column in input_data.data_frame.columns:
                if self.transaction.lmd['stats_v2'][column]['is_foreign_key']:
                    self.transaction.lmd['columns_to_ignore'].append(column)
                    cols_to_drop.append(column)
                    self.log.warning(f'Dropping column {column} because it is a foreign key')
            if cols_to_drop:
                input_data.data_frame.drop(columns=cols_to_drop, inplace=True)

        # Standartize data
        for column in input_data.data_frame.columns:
            data_type = self.transaction.lmd['stats_v2'][column]['typing']['data_type']
            data_subtype = self.transaction.lmd['stats_v2'][column]['typing']['data_subtype']

            if data_type == DATA_TYPES.NUMERIC:
                self._apply_to_all_data(input_data, column, _clean_float_or_none)
                self._apply_to_all_data(input_data, column, _handle_nan)

                if data_subtype == DATA_SUBTYPES.INT:
                    self._apply_to_all_data(input_data, column, _try_round)

            if data_type == DATA_TYPES.DATE:
                if data_subtype == DATA_SUBTYPES.DATE:
                    self._apply_to_all_data(input_data, column, _standardize_date)

                elif data_subtype == DATA_SUBTYPES.TIMESTAMP:
                    self._apply_to_all_data(input_data, column, _standardize_datetime)

            if data_type == DATA_TYPES.CATEGORICAL:
                self._apply_to_all_data(input_data, column, str)

            if data_type == DATA_TYPES.TEXT:
                self._apply_to_all_data(input_data, column, str)

            if self.transaction.hmd['model_backend'] == 'lightwood':
                if data_type == DATA_TYPES.DATE:
                    self._apply_to_all_data(input_data, column, _standardize_datetime)
                    self._apply_to_all_data(input_data, column, _lightwood_datetime_processing)
                    self._apply_to_all_data(input_data, column, _handle_nan)
