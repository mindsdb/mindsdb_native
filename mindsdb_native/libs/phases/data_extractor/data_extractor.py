from mindsdb_native.config import CONFIG
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.libs.helpers.text_helpers import hashtext
from mindsdb_native.external_libs.stats import calculate_sample_size

from pandas.api.types import is_numeric_dtype
import random
import traceback
import pandas as pd
import numpy as np


class DataExtractor(BaseModule):
    def _data_from_when(self):
        """
        :return:
        """
        when_conditions = self.transaction.hmd['when']

        when_conditions_list = []
        # here we want to make a list of the type  ( ValueForField1, ValueForField2,..., ValueForFieldN ), ...
        for when_condition in when_conditions:
            cond_list = [None] * len(self.transaction.lmd['columns'])  # empty list with blanks for values

            for condition_col in when_condition:
                col_index = self.transaction.lmd['columns'].index(condition_col)
                cond_list[col_index] = when_condition[condition_col]

            when_conditions_list.append(cond_list)

        result = pd.DataFrame(when_conditions_list, columns=self.transaction.lmd['columns'])

        return result

    def _data_from_when_data(self):
        df = self.transaction.hmd['when_data']
        df = df.where((pd.notnull(df)), None)

        for col in self.transaction.lmd['columns']:
            if col not in df.columns:
                df[col] = [None] * len(df)
        return df

    def _apply_sort_conditions_to_df(self, df):
        """

        :param df:
        :return:
        """

        # apply order by (group_by, order_by)
        if self.transaction.lmd['model_is_time_series']:
            asc_values = [order_tuple[ORDER_BY_KEYS.ASCENDING_VALUE] for order_tuple in self.transaction.lmd['model_order_by']]
            sort_by = [order_tuple[ORDER_BY_KEYS.COLUMN] for order_tuple in self.transaction.lmd['model_order_by']]

            if self.transaction.lmd['model_group_by']:
                sort_by = self.transaction.lmd['model_group_by'] + sort_by
                asc_values = [True for i in self.transaction.lmd['model_group_by']] + asc_values
            df = df.sort_values(sort_by, ascending=asc_values)

        elif self.transaction.lmd['type'] == TRANSACTION_LEARN:
            # if its not a time series, randomize the input data and we are learning
            df = df.sample(frac=1, random_state=len(df))

        return df


    def _get_prepared_input_df(self):
        """

        :return:
        """
        df = None

        # if transaction metadata comes with some data as from_data create the data frame
        if 'from_data' in self.transaction.hmd and self.transaction.hmd['from_data'] is not None:
            # make sure we build a dataframe that has all the columns we need
            df = self.transaction.hmd['from_data']
            df = df.where((pd.notnull(df)), None)

        if  self.transaction.lmd['type'] == TRANSACTION_PREDICT:
            if self.transaction.hmd['when_data'] is not None:
                df = self._data_from_when_data()
            else:
                # if no data frame yet, make one
                df = self._data_from_when()

        df = self._apply_sort_conditions_to_df(df)

        # Mutable lists -> immutable tuples
        # (lists caused TypeError: uhashable type 'list' in TypeDeductor phase)
        df = df.applymap(lambda cell: tuple(cell) if isinstance(cell, list) else cell)

        groups = df.columns.to_series().groupby(df.dtypes).groups
        
        # @TODO: Maybe move to data cleaner ? Seems kind of out of place here
        if np.dtype('datetime64[ns]') in groups:
            for colname in groups[np.dtype('datetime64[ns]')]:
                df[colname] = df[colname].astype(str)

        return df


    def _validate_input_data_integrity(self):
        """
        :return:
        """
        if self.transaction.input_data.data_frame.shape[0] <= 0:
            error = 'Input Data has no rows, please verify from_data or when_conditions'
            self.log.error(error)
            raise ValueError(error)

        if self.transaction.lmd['type'] == TRANSACTION_LEARN:
            for col_target in self.transaction.lmd['predict_columns']:
                if col_target not in self.transaction.input_data.columns:
                    err = 'Trying to predict column {column} but column not in source data'.format(column=col_target)
                    self.log.error(err)
                    self.transaction.error = True
                    self.transaction.errorMsg = err
                    raise ValueError(err)
                    return

    def _set_user_data_subtypes(self):
        if 'from_data' in self.transaction.hmd and self.transaction.hmd['from_data'] is not None:
            for col in self.transaction.hmd['from_data'].data_subtypes:
                self.transaction.lmd['data_types'][col] = self.transaction.hmd['from_data'].data_types[col]
                self.transaction.lmd['data_subtypes'][col] = self.transaction.hmd['from_data'].data_subtypes[col]

    def run(self):
        # --- Dataset gets randomized or sorted (if timeseries) --- #
        result = self._get_prepared_input_df()
        # --- Dataset gets randomized or sorted (if timeseries) --- #

        # --- Some information about the dataset gets transplanted into transaction level variables --- #
        self.transaction.input_data.columns = result.columns.values.tolist()
        self.transaction.lmd['columns'] = self.transaction.input_data.columns
        self.transaction.input_data.data_frame = result
        # --- Some information about the dataset gets transplanted into transaction level variables --- #

        self._set_user_data_subtypes()

        # --- Some preliminary dataset integrity checks --- #
        self._validate_input_data_integrity()
        # --- Some preliminary dataset integrity checks --- #
