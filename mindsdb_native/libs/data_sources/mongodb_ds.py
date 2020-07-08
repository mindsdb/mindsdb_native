import pandas as pd
from pymongo import MongoClient

from mindsdb_native.libs.data_types.data_source import DataSource


class MongoDS(DataSource):
    def _setup(self, collection, query=None, host='localhost', port=27017, user='admin',
               password='123', database='database'):
        
        if not isinstance(collection, str):
            raise TypeError('collection must be a str')
        
        if query is None:
            query = {}
        else:
            if not isinstance(query, dict):
                raise TypeError('query must be a dict')

        conn = MongoClient(host=host,
                           port=port,
                           username=user,
                           password=password)

        db = conn[database]
        coll = db[collection]

        df = pd.DataFrame(list(coll.find(query, {'_id': 0})))

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
