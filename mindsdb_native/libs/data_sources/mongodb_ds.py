import pandas as pd
from pymongo import MongoClient

from mindsdb_native.libs.data_types.data_source import DataSource

class MongoDS(DataSource):
    def __init__(self, query, collection, database='database',
                 host='localhost', port=27017, user='admin', password='123'):
        super().__init__()
        self._query = query
        self.collection = collection
        self.database = database
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password

    def query(self, q):
        if not isinstance(q, dict):
            raise TypeError('query must be a dict')

        conn = MongoClient(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password
        )

        db = conn[self.database]
        coll = db[self.collection]

        df = pd.DataFrame(list(coll.find(q, {'_id': 0})))

        return df, self._make_colmap(df)

    def name(self):
        return '{}: {}/{}'.format(
            self.__class__.__name__,
            self.database,
            self.collection
        )
