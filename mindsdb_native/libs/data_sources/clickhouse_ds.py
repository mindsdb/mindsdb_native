import pandas as pd
import requests

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_types.mindsdb_logger import log


class ClickhouseDS(DataSource):
    def __init__(self, query, host='localhost', user='default', password=None,
                 port=8123, protocol='http'):

        if protocol not in ('https', 'http'):
            raise ValueError('Unexpected protocol {}'.fomat(protocol))

        self.query = query
        self.host = host
        self.user = user
        self.password = password
        self.port = int(port)
        self.protocol = protocol

        if ' format ' in query.lower():
            raise Exception('Please refrain from adding a "FORMAT" statement to the query')

        super().__init__(sql_query=query)

    def _fix_query(self, query):
        return '{} FORMAT JSON;'.format(query.rstrip(" ;\n"))

    def _setup(self, query=None, **kwargs):
        if query is None:
            q = self._fix_query(self.query)
        else:
            q = self._fix_query(query)

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
            log.error(f'Got an invalid response from the database: {response.text}')
            raise Exception(response.text)
        
        df = pd.DataFrame(data)

        return self._make_col_map(df)

    def name(self):
        return '{}'.format(self.__class__.__name__)
