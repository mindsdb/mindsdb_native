import pandas as pd
from sqlite3 import connect

from mindsdb_native.libs.data_types.data_source import SQLDataSource

class SQLite3DS(SQLDataSource):
    def __init__(self, query, database):
        super().__init__(query=query)
        self.database = database

    def query(self, q):
        with connect(self.database) as con:
            df = pd.read_sql(q, con=con)
        return df, self._make_colmap(df)
    
    def name(self):
        return 'SQLite3 - {}'.format(self._query)