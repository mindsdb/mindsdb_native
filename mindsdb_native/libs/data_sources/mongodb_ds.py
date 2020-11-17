import pandas as pd
from pymongo import MongoClient

from mindsdb_native.libs.data_types.data_source import DataSource

class MongoDS(DataSource):
    def __init__(self, *args, **kwargs):
        self.is_sql = False
        super(MongoDS, self).__init__(*args, **kwargs)

    def _setup(self, collection, query=None, database='database',
               host='localhost', port=27017, user='admin', password='123'):

        if not isinstance(collection, str):
            raise TypeError('collection must be a str')

        self._database_name = database
        self._collection_name = collection

        if query is None:
            query = {}
        else:
            if not isinstance(query, dict):
                raise TypeError('query must be a dict')

        conn = MongoClient(
            host=host,
            port=int(port),
            username=user,
            password=password
        )

        db = conn[database]
        coll = db[collection]

        df = pd.DataFrame(list(coll.find(query, {'_id': 0})))

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map

    def name(self):
        return '{}: {}/{}'.format(
            self.__class__.__name__,
            self._database_name,
            self._collection_name
        )
