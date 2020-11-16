import pandas as pd
import mysql.connector

from mindsdb_native.libs.data_types.data_source import DataSource


class MariaDS(DataSource):
    def __init__(self, query, database='mysql', host='localhost', port=3306,
                 user='root', password=''):
        self.query = query
        self.database = database
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password
        super().__init__(sql_query=query)

    def _setup(self, query=None, **kwargs):
        con = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

        df = pd.read_sql(query or self.query, con=con)
        con.close()

        return super()._setup(df)

    def name(self):
        return '{}: {}'.format(
            self.__class__.__name__,
            self._database_name
        )
