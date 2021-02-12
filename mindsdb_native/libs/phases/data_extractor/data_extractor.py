import json
from copy import deepcopy

from mindsdb_native.config import CONFIG
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.libs.helpers.text_helpers import hashtext
from mindsdb_native.external_libs.stats import calculate_sample_size
from mindsdb_native.libs.helpers.query_composer import create_history_query

from pandas.api.types import is_numeric_dtype
import random
import traceback
import pandas as pd
import numpy as np

from mindsdb_native.libs.constants.mindsdb import (
    TRANSACTION_ANALYSE,
    TRANSACTION_LEARN
)
from mindsdb_native.libs.helpers.json_helpers import unnest_df


class DataExtractor(BaseModule):
    def _data_from_when_data(self, df):
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
        if self.transaction.lmd['tss']['is_timeseries']:
            asc_values = [True for _ in self.transaction.lmd['tss']['order_by']]
            sort_by = self.transaction.lmd['tss']['order_by']

            if self.transaction.lmd['tss']['group_by'] is not None:
                sort_by = self.transaction.lmd['tss']['group_by'] + sort_by
                asc_values = [True for _ in self.transaction.lmd['tss']['group_by']] + asc_values

            df = df.sort_values(sort_by, ascending=asc_values)

        # if its not a time series, randomize the input data and we are learning
        if not self.transaction.lmd['tss']['is_timeseries'] and self.transaction.lmd['type'] == TRANSACTION_LEARN:
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

        if self.transaction.lmd['type'] == TRANSACTION_PREDICT:
            if self.transaction.hmd['when_data'] is not None:
                df = self.transaction.hmd['when_data']
            else:
                df = pd.DataFrame(self.transaction.hmd['when'])
                df, _ = unnest_df(df)

            df = self._data_from_when_data(df)

            if self.transaction.lmd['setup_args'] is not None and self.transaction.lmd['tss']['is_timeseries'] and self.transaction.lmd['use_database_history']:
                self.log.warning('Using automatic database history sourcing, will be selecting rows from the same table you used to train the original model.')
                if 'make_predictions' not in df.columns:
                    df['make_predictions'] = [True] * len(df)

                setup_args = deepcopy(self.transaction.lmd['setup_args'])


                if len(df) > 1 and self.transaction.lmd['tss']['group_by'] is not None:
                    encountered_set = set()
                    unique_group_by_rows = []
                    for i in range(len(df)):
                        val_tuple = tuple()
                        for group_col in self.transaction.lmd['tss']['group_by']:
                            val_tuple = tuple([*val_tuple,df.iloc[i][group_col]])
                            if val_tuple not in encountered_set:
                                encountered_set.add(val_tuple)
                                unique_group_by_rows.append(df.iloc[i])
                else:
                    unique_group_by_rows = [df.iloc[0]]

                historical_df = None

                for row in unique_group_by_rows:
                    setup_args['query'] = create_history_query(setup_args['query'], self.transaction.lmd['tss'], self.transaction.lmd['stats_v2'], row)

                    if historical_df is None:
                        historical_df = self.transaction.hmd['from_data_type'](**setup_args).df
                    else:
                        historical_df = pd.concat(historical_df,self.transaction.hmd['from_data_type'](**setup_args).df)

                historical_df['make_predictions'] = [False] * len(historical_df)
                for col in historical_df.columns:
                    if df[col].iloc[0] is not None:
                        historical_df[col] = [type(df[col].iloc[0])(x) for x in historical_df[col]]

                df = pd.concat([df,historical_df])

        for col, f in self.transaction.lmd.get('apply_to_columns', {}).items():
            df.loc[:, col] = df[col].apply(f)

        # Sorting here *should* only be needed at learn time
        if self.transaction.lmd['type'] == TRANSACTION_LEARN:
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
                if col_target not in self.transaction.lmd['columns']:
                    err = 'Trying to predict column {column} but column not in source data'.format(column=col_target)
                    self.log.error(err)
                    self.transaction.error = True
                    self.transaction.errorMsg = err
                    raise ValueError(err)

    def _set_user_data_subtypes(self):
        if 'from_data' in self.transaction.hmd and self.transaction.hmd['from_data'] is not None:
            for col in self.transaction.hmd['from_data'].data_subtypes:
                self.transaction.lmd['data_types'][col] = self.transaction.hmd['from_data'].data_types[col]
                self.transaction.lmd['data_subtypes'][col] = self.transaction.hmd['from_data'].data_subtypes[col]

    def _count_isna(self, df):
        count = 0
        for col in df.columns:
            count += df[col].isna().sum()
        return count

    def run(self):
        if self.transaction.hmd.get('from_data') is not None:
            self.transaction.lmd['data_source_name'] = self.transaction.hmd['from_data'].name()

        # --- Dataset gets randomized or sorted (if timeseries) --- #
        df = self._get_prepared_input_df()
        # --- Dataset gets randomized or sorted (if timeseries) --- #

        # --- Replace -inf/inf values with None --- #
        null_count_1 = self._count_isna(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        null_count_2 = self._count_isna(df)
        inf_count = (null_count_2 - null_count_1)
        if inf_count > 0:
            self.log.warning('Your dataset contains {} -inf/inf values, replacing them with None'.format(inf_count))
        # --- Replace -inf/inf values with None --- #

        # --- Some information about the dataset gets transplanted into transaction level variables --- #
        self.transaction.lmd['columns'] = [x for x in df.columns.values.tolist() if x != 'make_predictions']
        self.transaction.input_data.data_frame = df
        # --- Some information about the dataset gets transplanted into transaction level variables --- #

        self._set_user_data_subtypes()

        # --- Some preliminary dataset integrity checks --- #
        self._validate_input_data_integrity()
        # --- Some preliminary dataset integrity checks --- #
