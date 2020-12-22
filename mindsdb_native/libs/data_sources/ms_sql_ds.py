import pandas as pd
import pymssql
from mindsdb_native.libs.data_types.data_source import SQLDataSource


class MSSQLDS(SQLDataSource):
    def __init__(self, query, database='master', host='localhost',
               port=1433, user='sa', password=''):
        super().__init__(query=query)
        self.database = database
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password

    def query(self, q):
        with pymssql.connect(server=self.host, host=self.host, user=self.user, password=self.password, database=self.database, port=self.port) as con:
            df = pd.read_sql(q, con=con)
        return df, self._make_colmap(df)

    def name(self):
        return 'Microsoft SQL - {}'.format(self._query)
