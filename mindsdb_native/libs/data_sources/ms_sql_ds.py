import pandas as pd
import pyodbc
from mindsdb_native.libs.data_types.data_source import DataSource


def mssql_connect(host, port, user, password, database):
    return pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+host+','+str(port)+';DATABASE='+database+';UID='+user+';PWD='+password)


class MSSQLDS(DataSource):
    def _setup(self, query=None, host='.', user='sa', password='',
               database='mssql', port=1433, table=None):

        if query is None:
            query = f'SELECT * FROM {table}'

        con = mssql_connect(host=host, port=port, user=user, password=password, database=database)

        df = pd.read_sql(query, con=con)
        con.close()

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map
