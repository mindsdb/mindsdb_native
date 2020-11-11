import pandas as pd
import requests

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_types.mindsdb_logger import log


class ClickhouseDS(DataSource):
    def __init__(self, *args, **kwargs):
        self.is_sql = True
        super(ClickhouseDS, self).__init__(*args, **kwargs)

    def _setup(self, query, host='localhost', user='default', password=None,
               port=8123, protocol='http'):
        if protocol not in ('https', 'http'):
            raise ValueError('Unexpected protocol {}'.fomat(protocol))

        if ' format ' in query.lower():
            err_msg = 'Please refrain from adding a "FORMAT" statement to the query'
            log.error(err_msg)
            raise Exception(err_msg)

        self.setup_args = {
            'query' : query
            ,'host' : host
            ,'user' : user
            ,'password' : password
            ,'port' : port
            ,'protocol' : protocol
        }

        query = '{} FORMAT JSON'.format(query.rstrip(" ;\n"))
        log.info(f'Getting data via the query: "{query}"')

        params = {'user': user}
        if password is not None:
            params['password'] = password

        response = requests.post(
            f'{protocol}://{host}:{port}',
            data=query,
            params=params
        )

        try:
            data = response.json()['data']
        except Exception:
            log.error(f'Got an invalid response from the database: {response.text}')
            raise Exception(response.text)

        df = pd.DataFrame(data)

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map

    def name(self):
        return '{}'.format(self.__class__.__name__)
