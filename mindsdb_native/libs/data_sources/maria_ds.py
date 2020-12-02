import pandas as pd
import mysql.connector

from mindsdb_native.libs.data_types.data_source import SQLDataSource


class MariaDS(SQLDataSource):
    def __init__(self, query, database='mysql', host='localhost',
                 port=3306, user='root', password=''):
        super().__init__(query)
        self.database = database
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password

    def query(self, q):
        con = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
        df = pd.read_sql(q, con=con)
        con.close()

        return df, self._make_colmap(df)

    def name(self):
        return 'MariaDB - {}'.format(self._query)
