import os

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_sources.file_ds import FileDS
from pandas import DataFrame

from mindsdb_native.libs.data_types.mindsdb_logger import log


def getDS(from_data):
    '''
    Get a datasource give the input

    :param input: a string or an object
    :return: a datasource
    '''
    if isinstance(from_data, DataSource):
        return from_data
    elif isinstance(from_data, DataFrame):
        return DataSource(from_data)
    elif isinstance(from_data, str):
        if os.path.isfile(from_data) or from_data.startswith('http:') or from_data.startswith('https:'):
            return FileDS(from_data)
    raise ValueError('from_data must be one of: [DataSource, DataFrame, file path, file URL]')
