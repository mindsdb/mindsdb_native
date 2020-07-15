import pandas as pd
import mysql.connector

from mindsdb_native.libs.data_types.data_source import DataSource


class MariaDS(DataSource):

    def _setup(self, table, query=None, database='mysql', host='localhost',
               port=3306, user='root', password=''):

        if query is None:
            query = f'SELECT * FROM {table}'

        con = mysql.connector.connect(host=host,
                                      port=port,
                                      user=user,
                                      password=password,
                                      database=database)
        df = pd.read_sql(query, con=con)
        con.close()

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
