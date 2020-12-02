import pandas as pd
import requests

from mindsdb_native.libs.data_types.data_source import SQLDataSource
from mindsdb_native.libs.data_types.mindsdb_logger import log


class ClickhouseDS(SQLDataSource):
    def __init__(self, query, host='localhost', user='default', password=None,
                 port=8123, protocol='http'):

        if ' format ' in query.lower():
            raise Exception('Please refrain from adding a "FORMAT" statement to the query')

        super().__init__(query)

        self.host = host
        self.user = user
        self.password = password
        self.port = int(port)
        self.protocol = protocol

        if protocol not in ('https', 'http'):
            raise ValueError('Unexpected protocol {}'.fomat(protocol))

    def query(self, q):
        q = '{} FORMAT JSON'.format(q.rstrip(" ;\n"))
        params = {'user': self.user}
        if self.password is not None:
            params['password'] = self.password

        response = requests.post(
            f'{self.protocol}://{self.host}:{self.port}',
            data=q,
            params=params
        )

        try:
            data = response.json()['data']
        except Exception:
            raise Exception('Got an invalid response from the database: {response.text}')

        df = pd.DataFrame(data)

        return df, self._make_colmap(df)

    def name(self):
        return 'Clickhouse - {}'.format(self._query)
