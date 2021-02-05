from copy import deepcopy

import pandas as pd
import moz_sql_parser
from moz_sql_parser.keywords import binary_ops
import traceback

from mindsdb_native.libs.constants.mindsdb import (
    DATA_TYPES_SUBTYPES,
    DATA_TYPES,
    DATA_SUBTYPES
)
from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.libs.helpers.json_helpers import unnest_df

def unnest(df, col_map):
    df, unnested = unnest_df(df)
    if unnested > 0:
        col_map = {}
        for col in df.columns:
            col_map[col] = col
    return df, col_map

class DataSource:
    def __init__(self, df=None, query=None):
        if type(self) is DataSource and df is None:
            raise Exception('When you\'re instantiating DataSource, you must provide :param df:')

        self._query = query

        if df is not None:
            self._internal_df = df
            self._internal_col_map = self._make_colmap(df)
            self._internal_df, self._internal_col_map = unnest(self._internal_df, self._internal_col_map)
        else:
            self._internal_df = None
            self._internal_col_map = None

        self.data_types = {}
        self.data_subtypes = {}

    def __len__(self):
        return len(self.df)

    def _make_colmap(self, df):
        col_map = {}
        for col in df.columns:
            col_map[col] = col
        return col_map

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

    def _extract_and_map(self):
        if self._internal_df is None:
            self._internal_df, self._internal_col_map = self.query(self._query)
            self._internal_df, self._internal_col_map = unnest(self._internal_df, self._internal_col_map)

    @property
    def df(self):
        self._extract_and_map()
        return self._internal_df

    @df.setter
    def df(self, df):
        self._internal_df = df

    @property
    def _col_map(self):
        self._extract_and_map()
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

    def query(self, q=None):
        """
        :param q: a query specific to type of datasource
        Datasources must override this method to return pandas.DataFrame
        based on :param q:
        e.g. for MySqlDS :param q: must be a SQL query
             for MongoDS :param q: must be a dictionary

        :return: tuple(pandas.DataFrame, dict)
        """

        # If it's not a subclass of DataSource, then dataframe was provided
        # in the constructor and we just return it
        if type(self) is DataSource:
            return self._internal_df, self._internal_col_map

        # If it is a subclass of DataSource, then this method must be overriden
        else:
            raise NotImplementedError('You must override DataSource.query')

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
            df = df[df[col].apply(str).str.contains(str(val).replace("%", ""))]
        if cond == '=':
            df = df[( df[col] == val ) | ( df[col] == str(val) )]
        if cond == '!=':
            df = df[( df[col] != val ) & ( df[col] != str(val) )]

        return df

    def filter(self, where=None, limit=None, get_col_map=False):
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
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            return getattr(self.df, attr)

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

    def name(self):
        return 'DataFrame'


class SQLDataSource(DataSource):
    def __init__(self, query):
        super().__init__(query=query)

    def filter(self, where=None, limit=None, get_col_map=False):
        try:
            parsed_query = moz_sql_parser.parse(self._query.replace('FORMAT JSON', ''))

            modified_columns = []
            for col, op, value in where or []:
                past_where_clause = parsed_query.get('where', {})

                op = op.lower()
                op_json = binary_ops.get(op, None)

                if op_json is None:
                    log.warning(f"Operator: {op} not found in the sql parser operator list\n Using it anyway.")
                    op_json = op

                if op == 'like':
                    value = '%' + value.strip('%') + '%'
                    if 'clickhouse' in self.name().lower():
                        col = f'toString({col})'
                    elif 'postgres' in self.name().lower():
                        col = f'{col}::text'
                    elif 'mariadb' in self.name().lower() or 'mysql' in self.name().lower() or 'mssql' in self.name().lower():
                        col = f'CAST({col} AS TEXT)'

                modified_columns.append(col)

                where_clause = {op_json: [col, value]}

                if len(past_where_clause) > 0:
                    where_clause = {'and': [where_clause, past_where_clause]}

                parsed_query['where'] = where_clause

            if limit is not None:
                parsed_query['limit'] = limit

            query = moz_sql_parser.format(parsed_query)
            query = query.replace('"', "'")
            query = query.replace("'.'",".")

            for col in modified_columns:
                if f"'{col}'" in query:
                    query = query.replace(f"'{col}'", col)

            df, col_map = self.query(query)
            df, col_map = unnest(df, col_map)
            if get_col_map:
                return df, col_map
            else:
                return df

        except Exception as e:
            print(traceback.format_exc())
            print('Failed to filter using SQL: ', e)
            return super().filter(where=where, limit=limit, get_col_map=get_col_map)

    @property
    def _col_map(self):
        if self._internal_col_map is None:
            _, self._internal_col_map = self.filter(where=[], limit=200, get_col_map=True)
        return self._internal_col_map

    def name(self):
        raise NotImplementedError
