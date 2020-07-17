import six
import numpy as np
from mindsdb_native.libs.phases.base_module import BaseModule


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

    def _convert_iterables_to_tuples(self, df):
        for column in df.dropna().select_dtypes(include='object').columns:
            test_example = df[column].iloc[0]
            is_iterable = False
            if not isinstance(test_example, six.string_types):
                try:
                    iter(test_example)
                    # It's an iterable: array, tuple, list or something else
                    is_iterable = True
                except Exception:
                    # It's not an iterable, ignore
                    pass

            if is_iterable:
                # Convert all to tuples
                df[column] = df[column].apply(lambda iterable: tuple(iterable) if iterable is not None else iterable)

    def run(self):
        df = self.transaction.input_data.data_frame

        empty_columns = self._get_empty_columns(df)
        self.transaction.lmd['empty_columns'] = empty_columns
        self.transaction.lmd['columns_to_ignore'] += empty_columns
        df.drop(columns=self.transaction.lmd['columns_to_ignore'], inplace=True)

        self._remove_missing_targets(df)

        if self.transaction.lmd.get('deduplicate_data'):
            self._remove_duplicate_rows(df)

        self._convert_iterables_to_tuples(df)

        self.transaction.input_data.data_frame = df
