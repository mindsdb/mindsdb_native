import os

import pandas as pd
import pg8000

from mindsdb_native.libs.data_types.data_source import SQLDataSource


class PostgresDS(SQLDataSource):
    def __init__(self, query, database='postgres', host='localhost',
                 port=5432, user='postgres', password=''):
        super().__init__(query)
        self.database = database
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password

    def query(self, q):
        con = pg8000.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

        df = pd.read_sql(q, con=con)
        con.close()

        df.columns = [x if isinstance(x, str) else x.decode('utf-8') for x in df.columns]
        for col_name in df.columns:
            try:
                df[col_name] = df[col_name].apply(lambda x: x if isinstance(x, str) else x.decode('utf-8'))
            except Exception:
                pass

        return df, self._make_colmap(df)

    def name(self):
        return 'Postgres - {}'.format(self._query)
