import os

import pandas as pd
import pg8000

from mindsdb_native.libs.data_types.data_source import SQLDataSource


class RedshiftDS(SQLDataSource):
    def __init__(self, query, database='dev', host='localhost',
                 port=5439, user='awsuser', password=''):
        super().__init__(query=query)
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

        for col_name in df.columns:
            try:
                df[col_name] = df[col_name].apply(lambda x: x.decode("utf-8"))
            except Exception:
                pass

        return df, self._make_colmap(df)

    def name(self):
        return 'Redshift - {}'.format(self._query)
