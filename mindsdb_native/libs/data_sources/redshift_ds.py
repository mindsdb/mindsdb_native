import os

import pandas as pd 
import pg8000 

from mindsdb_native.libs.data_types.data_source import DataSource


class RedshiftDS(DataSource):
    
    def _setup(self, table=None, query=None, database='dev', host='localhost',
               port=5439, user='awsuser', password=''):

        self._database_name = database
        self._table_name = table

        if query is None:
            query = f'SELECT * FROM {table}'

        con = pg8000.connect(database=database, user=user, password=password, host=host, port=port)
        df = pd.read_sql(query, con=con)
        con.close()

        for col_name in df.columns:
            try:
                df[col_name] = df[col_name].apply(lambda x: x.decode("utf-8"))
            except:
                pass

        col_map = {}
        for col in df.columns:
            col_map[col] = col
        
        return df, col_map

    def name(self):
        return '{}: {}/{}'.format(
            self.__class__.__name__,
            self._database_name,
            self._table_name
        )