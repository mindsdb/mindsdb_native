import os

import pandas as pd
import pg8000

from mindsdb_native.libs.data_types.data_source import DataSource


class PostgresDS(DataSource):

    def _setup(self, query=None, host='localhost', user='postgres', password='', database='postgres', port=5432, table=None):

        if query is None:
            query = f'SELECT * FROM {table}'

        con = pg8000.connect(database=database, user=user, password=password,
                             host=host, port=port)
        df = pd.read_sql(query, con=con)
        con.close()

        df.columns = [x.decode('utf-8') for x in df.columns]
        for col_name in df.columns:
            try:
                df[col_name] = df[col_name].apply(lambda x: x.decode("utf-8"))
            except:
                pass

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
