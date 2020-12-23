import re

import pandas as pd
from pymongo import MongoClient

from mindsdb_native.libs.data_types.data_source import DataSource

class MongoDS(DataSource):
    def __init__(self, query, collection, database='database',
                 host='127.0.0.1', port=None, user=None, password=None):
        super().__init__()

        if not isinstance(query, dict):
            raise TypeError('query must be a dict')
        else:
            self._query = query

        if not re.match(r'^mongodb(\+srv)?:\/\/', host.lower()):
            port = int(port or 27017)

        self.collection = collection
        self.database = database
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def query(self, q):
        assert isinstance(q, dict)

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
        return 'MongoDB - {}'.format(self._query)
