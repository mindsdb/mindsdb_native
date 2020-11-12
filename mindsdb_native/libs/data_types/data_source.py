from copy import deepcopy

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
        self._internal_df = None
        self._internal_col_map = None
        self.args = args
        self.kwargs = kwargs

        if 'is_sql' not in self.__dict__:
            self.is_sql = False

        # @TOOD Let's just make the query an "obligatory" first arg of all sql datasources
        if self.is_sql:
            # @TODO Also, even if we do the above, it might be wiser to set the `query` inside each separate datasource child ?
            try:
                self.query = self.kwargs['query']
            except Exception as e:
                self.query = self.args[0]

        self._cleanup()

    def __len__(self):
        return len(self.df)

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

    @property
    def df(self):
        if self._internal_df is None:
            self._internal_df, self._internal_col_map = self._setup(*self.args, **self.kwargs)
        return self._internal_df

    @df.setter
    def df(self, df):
        self._internal_df = df

    @property
    def _col_map(self):
        if self._internal_col_map is None:
            # Probably more elegant without the `if` but python is dumb and can easily get itself into weird internal loops => core dumps if
            if self.is_sql:
                _, self._internal_col_map = self.filter(where=[], limit=1, get_col_map=True)
            else:
                self._internal_df, self._internal_col_map = self._setup(*self.args, **self.kwargs)

        return self._internal_col_map

    @_col_map.setter
    def col_map(self, _col_map):
        self._internal_col_map = _col_map

    def _set_df(self, df, col_map):
        self._internal_df = df
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

        self._internal_df.drop(columns=columns_to_drop, inplace=True)

    def _filter_df(self, raw_condition, df):
        """Convert filter conditions to a paticular
        DataFrame instance"""
        col, cond, val = raw_condition
        cond = cond.lower()
        df = df[df[col].notnull()]

        if cond == '>':
            df = df[pd.to_numeric(df[col], errors='coerce') > val]
        if cond == '<':
            df = df[pd.to_numeric(df[col], errors='coerce') < val]
        if cond == 'like':
            df = df[df[col].str.contains(str(val).replace("%", ""))]
        if cond == '=':
            df = df[( df[col] == val ) | ( df[col] == str(val) )]
        if cond == '!=':
            df = df[( df[col] != val ) & ( df[col] != str(val) )]

        return df

    def filter(self, where=None, limit=None, get_col_map=False):
        """Convert SQL like filter requests to pandas DataFrame filtering"""
        try:
            assert self.is_sql
            parsed_query = moz_sql_parser.parse(self.query)

            for col, op, value in where or []:
                past_where_clause = parsed_query.get('where', {})

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
            query = query.replace('"',"'")
            args = deepcopy(self.args)
            kwargs = deepcopy(self.kwargs)

            if 'query' in kwargs:
                kwargs['query'] = query
            else:
                args[0] = query

            if get_col_map:
                return self._setup(*args, **kwargs)
            else:
                return self._setup(*args, **kwargs)[0]
        except Exception as e:
            df = self.df
            if where:
                for cond in where:
                    df = self._filter_df(cond, df)
            return df.head(limit) if limit else df

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getattr__(self, attr):
        """
        Map all other functions to the DataFrame
        """
        if attr == 'df':
            return self.df
        if attr == 'col_map':
            return self.col_map

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
