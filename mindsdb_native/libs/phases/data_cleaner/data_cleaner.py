import numpy as np

from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.constants.mindsdb import TRANSACTION_LEARN


class DataCleaner(BaseModule):
    def _get_empty_columns(self, df):
        empty_columns = []
        for col_name in df.columns.values:
            if len(df[col_name].dropna()) < 1:
                empty_columns.append(col_name)
                self.log.warning(f'Column "{col_name}" is empty ! We\'ll go ahead and ignore it, please make sure you gave mindsdb the correct data.')
        return empty_columns

    def _remove_missing_targets(self, df):
        initial_len = len(df)
        df.dropna(subset=self.transaction.lmd['predict_columns'], inplace=True)
        no_dropped = initial_len - len(df)
        if no_dropped > 0:
            self.log.warning(
                f'Dropped {no_dropped} rows because they had null values in one or more of the columns that we are trying to predict. Please always provide non-null values in the columns you want to predict !')

    def _remove_duplicate_rows(self, df):
        initial_len = len(df)
        df.drop_duplicates(inplace=True)
        no_dropped = initial_len - len(df)
        if no_dropped > 0:
            self.log.warning(f'Dropped {no_dropped} duplicate rows.')

    def run(self):
        df = self.transaction.input_data.data_frame

        for col, nulls in self.transaction.lmd.get('null_values', {}).items():
            # NOTE: df[col].replace(nulls, None, inplace=True) will
            # not work as (I) expected, here's an example of what it does:
            #
            # >>> import pandas as pd
            # >>> df = pd.DataFrame({'my_column': [1, 2, 3, 'replace_me']})
            # >>> df['my_column'].replace('replace_me', None, inplace=True)
            # >>> df
            # my_column
            # 0         1
            # 1         2
            # 2         3
            # 3         3  <--- None is expected instead of 3
            #
            # That's why np.nan is used here instead of None
            df[col].replace(nulls, np.nan, inplace=True)

        empty_columns = self._get_empty_columns(df)

        self.transaction.lmd['empty_columns'] = empty_columns

        self.transaction.lmd['columns_to_ignore'].extend(
            set(empty_columns).difference(
                self.transaction.lmd['predict_columns']
            ).difference(
                self.transaction.lmd['force_column_usage']
            )
        )

        cols_to_drop = [col for col in df.columns if col in self.transaction.lmd['columns_to_ignore']]
        if len(cols_to_drop) > 0:
            df.drop(columns=cols_to_drop, inplace=True)

        if self.transaction.lmd.get('remove_columns_with_missing_targets', True):
            self._remove_missing_targets(df)

        len_before_dedupe = len(df)
        if self.transaction.lmd.get('deduplicate_data'):
            self._remove_duplicate_rows(df)
        len_after_dedupe = len(df)

        if len_after_dedupe < len_before_dedupe / 2:
            self.log.warning(f'Less than half of initial rows remain after deduplication. Consider passing `deduplicate_data=False` if training results are sub-par.')

        if self.transaction.lmd['type'] == TRANSACTION_LEARN:
            # Remove rows that only contain nulls
            df.dropna(axis=0, how='all', inplace=True)

            MINIMUM_ROWS = 10
            if len(df) < MINIMUM_ROWS:
                raise Exception('Your data contains very few rows ({}). The minimum is {} rows.'.format(
                    len(df),
                    MINIMUM_ROWS
                ))

            # remove target outlier rows based on z-score
            if self.transaction.lmd['remove_target_outliers'] != 0:
                for target in self.transaction.lmd['predict_columns']:
                    z_threshold = self.transaction.lmd['remove_target_outliers']
                    mean = df[target].mean()
                    sd = df[target].std()
                    df['scores'] = (df[target] - mean) / sd
                    df = df[df['scores'] < z_threshold]
                    df.pop('scores')

        self.transaction.input_data.data_frame = df
