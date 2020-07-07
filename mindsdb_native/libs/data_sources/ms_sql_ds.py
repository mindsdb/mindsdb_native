import pandas as pd
import pymssql
from mindsdb_native.libs.data_types.data_source import DataSource


class MSSQLDS(DataSource):
    def _setup(self, query=None, host='.', user='sa', password='',
               database='mssql', port=1433, table=None):

        if query is None:
            query = f'SELECT * FROM {table}'

        con = pymssql.connect(server=host, user=user,
                              password=password, database=database)  

        df = pd.read_sql(query, con=con)
        con.close()

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
