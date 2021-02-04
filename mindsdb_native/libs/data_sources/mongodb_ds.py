import re

import certifi
import pandas as pd
from pymongo import MongoClient
from pandas.api.types import is_numeric_dtype

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

        kwargs = {}

        if isinstance(self.user, str) and len(self.user) > 0:
            kwargs['username'] = self.user

        if isinstance(self.password, str) and len(self.password) > 0:
            kwargs['password'] = self.password

        if re.match(r'\/\?.*tls=true', self.host.lower()):
            kwargs['tls'] = True

        if re.match(r'\/\?.*tls=false', self.host.lower()):
            kwargs['tls'] = False

        if re.match(r'.*\.mongodb.net', self.host.lower()):
            kwargs['tlsCAFile'] = certifi.where()
            if kwargs.get('tls', None) is None:
                kwargs['tls'] = True

        conn = MongoClient(
            host=self.host,
            port=self.port,
            **kwargs
        )

        db = conn[self.database]
        coll = db[self.collection]

        df = pd.DataFrame(list(coll.find(q, {'_id': 0})))
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df, self._make_colmap(df)

    def name(self):
        return 'MongoDB - {}'.format(self._query)
