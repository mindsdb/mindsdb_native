from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.data_types.mindsdb_logger import log


class DataCleaner(BaseModule):
    def get_empty_columns(self, df):
        empty_columns = []
        for col_name in df.columns.values:
            if len(df[col_name].dropna()) < 1:
                empty_columns.append(col_name)
                self.log.warning(f'Column "{col_name}" is empty ! We\'ll go ahead and ignore it, please make sure you gave mindsdb the correct data.')

        return empty_columns

    def remove_missing_targets(self, df):
        initial_len = len(df)
        df.dropna(subset=self.transaction.lmd['predict_columns'], inplace=True)
        no_dropped = len(df) - initial_len
        if no_dropped > 0:
            self.log.warning(
                f'Dropped {no_dropped} rows because they had null values in one or more of the columns that we are trying to predict. Please always provide non-null values in the columns you want to predict !')
        return df

    def run(self):
        df = self.transaction.input_data.data_frame

        empty_columns = self.get_empty_columns(df)
        self.transaction.lmd['empty_columns'] = empty_columns
        self.transaction.lmd['columns_to_ignore'] += empty_columns
        df.drop(columns=self.transaction.lmd['columns_to_ignore'], inplace=True)

        df = self.remove_missing_targets(df)

        self.transaction.input_data.data_frame = df
