import pandas as pd
from pymongo import MongoClient

from mindsdb_native.libs.data_types.data_source import DataSource


class MongoDS(DataSource):
    def _setup(self, host='localhost', port=27017, user='admin', password='123',
               database='database', collection='collection'):

        conn = MongoClient(host=host,
                           port=port,
                           username=user,
                           password=password)

        db = conn[database]
        collection = db[collection]

        df = pd.DataFrame(list(collection.find({}, {'_id': 0})))

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
