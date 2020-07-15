import pandas as pd
import pytds
from mindsdb_native.libs.data_types.data_source import DataSource


class MSSQLDS(DataSource):
    def _setup(self, table, query=None, database='master', host='localhost',
               port=1433, user='sa', password=''):

        if query is None:
            query = f'SELECT * FROM {table}'

        with pytds.connect(dsn=host,
                           user=user,
                           password=password,
                           database=database) as con:
            df = pd.read_sql(query, con=con)
            
        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
