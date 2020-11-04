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
        df, col_map = self._setup(*args, **kwargs)
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
