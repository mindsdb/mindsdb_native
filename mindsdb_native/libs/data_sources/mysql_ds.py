import pandas as pd
import mysql.connector

from mindsdb_native.libs.data_types.data_source import DataSource


class MySqlDS(DataSource):
    def _setup(self, table=None, query=None, database='mysql', host='localhost',
               port=3306, user='root', password=''):

        self._database_name = database
        self._table_name = table

        if query is None:
            query = f'SELECT * FROM {table}'

        con = mysql.connector.connect(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database
        )
        df = pd.read_sql(query, con=con)
        con.close()

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map

    def name(self):
        return '{}: {}'.format(
            self.__class__.__name__,
            self._database_name
        )