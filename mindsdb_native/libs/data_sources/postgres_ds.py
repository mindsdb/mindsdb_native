import os

import pandas as pd
import pg8000

from mindsdb_native.libs.data_types.data_source import DataSource


class PostgresDS(DataSource):
    def __init__(self, query, database='postgres', host='localhost',
                 port=5432, user='postgres', password=''):
        self.query = query
        self.database = database
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        super().__init__(sql_query=query)

    def _setup(self, query=None, **kwargs):
        con = pg8000.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        df = pd.read_sql(query or self.query, con=con)
        con.close()

        df.columns = [x if isinstance(x, str) else x.decode('utf-8') for x in df.columns]
        for col_name in df.columns:
            try:
                df[col_name] = df[col_name].apply(lambda x: x if isinstance(x, str) else x.decode('utf-8'))
            except Exception:
                pass
        
        return self._make_col_map(df)

    def name(self):
        return '{}: {}/{}'.format(
            self.__class__.__name__,
            self._database_name,
            self._table_name
        )
