import re

import certifi
import pandas as pd
from pymongo import MongoClient
from pandas.api.types import is_numeric_dtype

from mindsdb_native.libs.data_types.data_source import DataSource

class MongoDS(DataSource):
    def __init__(self, query, collection, database='database',
                 host='127.0.0.1', port=None, user=None, password=None,
                 limit=None, sort_by=None):

        """params:
            limit: limit of requested documents
            sort_by: sort settings. it is a dict with fields as keys
                and '1'(ascending) '-1'(descending) as values
                example sort_by={'id': 1, 'vendor_id': -1}"""
        super().__init__()

        if not isinstance(query, dict):
            raise TypeError('query must be a dict')
        self._query = query

        if not re.match(r'^mongodb(\+srv)?:\/\/', host.lower()):
            port = int(port or 27017)

        self.collection = collection
        self.database = database
        self.limit = limit
        self.sort_by = sort_by
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

        raw_data = coll.find(q, {'_id': 0})
        if self.sort_by:
            raw_data = raw_data.sort(list((k, self.sort_by[k]) for k in self.sort_by))
        if self.limit:
            raw_data = raw_data.limit(self.limit)

        df = pd.DataFrame(list(raw_data))
        for col in df.columns:
            if not is_numeric_dtype(df[col]) or isinstance(df[col], dict):
                df[col] = df[col].astype(str)

        return df, self._make_colmap(df)

    def name(self):
        return 'MongoDB - {}'.format(self._query)
