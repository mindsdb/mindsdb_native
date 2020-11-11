import pandas as pd
import moz_sql_parser
from moz_sql_parser.keywords import binary_ops

from mindsdb_native.libs.constants.mindsdb import (
    DATA_TYPES_SUBTYPES,
    DATA_TYPES,
    DATA_SUBTYPES
)
from mindsdb_native.libs.data_types.mindsdb_logger import log


class DataSource:
    def __init__(self, *args, **kwargs):
        self.data_types = {}
        self.data_subtypes = {}
        self.is_dynamic = False
        df, col_map = self._setup(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self._set_df(df, col_map)
        self._cleanup()

    def __len__(self):
        return len(self.df)

    def name(self):
        return 'Unknown'

    def _setup(self, df, **kwargs):
        col_map = {}

        for col in df.columns:
            col_map[col] = col

        return df, col_map

    def _cleanup(self):
        pass

    def set_subtypes(self, data_subtypes):
        """
        :param data_subtypes: dict
        """
        for col, subtype in data_subtypes.items():
            if col not in self._col_map:
                log.warning(f'Column {col} not present in your data, ignoring the "{subtype}" subtype you specified for it')
                continue

            for type_, type_subtypes in DATA_TYPES_SUBTYPES.items():
                if subtype in type_subtypes:
                    self.data_types[col] = type_
                    self.data_subtypes[col] = subtype
                    break
            else:
                raise ValueError(f'Invalid data subtype: {subtype}')

    def _set_df(self, df, col_map):
        self.df = df
        self._col_map = col_map

    def drop_columns(self, column_list):
        """
        Drop columns by original names

        :param column_list: a list of columns that you want to drop
        """
        columns_to_drop = []

        for col in column_list:
            if col not in self._col_map:
                columns_to_drop.append(col)
            else:
                columns_to_drop.append(self._col_map[col])

        self.df.drop(columns=columns_to_drop, inplace=True)

    def _filter_to_pandas(self, raw_condition):
        """Convert filter conditions to a paticular
        DataFrame instance"""
        mapping = {
                    ">": lambda x, y: self._df[x] > y,
                    "LIKE": lambda x, y: self._df[x].str.contains(y.replace("%", "")),
                    "<": lambda x, y: self._df[x] < y,
                    "=": lambda x, y: self._df[x] == y,
                    "!=": lambda x, y: self._df[x] != y
                  }
        col, cond, val = raw_condition
        return mapping[cond](col, val)

    def filter(self, where=None, limit=None):
        """Convert SQL like filter requests to pandas DataFrame filtering"""
        if self.is_dynamic:
            parsed_query = moz_sql_parser.parse(self.query)

            for col, op, value in where or []:
                past_where_clause = self.parsed.get('where', {})

                op = op.lower()
                op_json = binary_ops.get(op, None)
                if op_json is None:
                    log.warning(f"Operator: {op} not found in: QueryBuilder._OPERATORS\n Using it anyway.")
                    op_json = op.lower()

                if op.lower() == 'like':
                    value = '%' + value.strip('%') + '%'

                where_clause = {op_json: [col, value]}

                if len(past_where_clause) > 0:
                    where_clause = {'and': [where_clause, past_where_clause]}

                parsed_query['where'] = where_clause

            if limit is not None:
                parsed_query['limit'] = limit

            query = moz_sql_parser.format(parsed_query)
            
            return self._setup(*self.args, query=query, **self.kwargs)._df
        else:
            if where:
                for cond in [self._filter_to_pandas(x) for x in where]:
                    self._df = self._df[cond]
            return self._df.head(limit) if limit else self._df

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, attr):
        """
        Map all other functions to the DataFrame
        """
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        else:
            return self.df.__getattr__(attr)

    def __getitem__(self, key):
        """
        Map all other items to the DataFrame
        """
        return self.df.__getitem__(key)

    def __setitem__(self, key, value):
        """
        Support item assignment, mapped to DataFrame
        """
        self.df.__setitem__(key, value)
