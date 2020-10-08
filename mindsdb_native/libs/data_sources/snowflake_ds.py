
import os

from snowflake import connector

from mindsdb_native.libs.data_types.data_source import DataSource


class SnowflakeDS(DataSource):

    def _setup(self, query, host, user, password, account, warehouse, database, schema, protocol='https', port=443):
        con = connector.connect(
                  host=host,
                  user=user,
                  password=password,
                  account=account,
                  warehouse=warehouse,
                  database=database,
                  schema=schema,
                  protocol='https',
                  port=port)
        # Create a cursor object.
        cur = con.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()

        cur.close()
        con.close()

        self._database = database
        self._warehouse = warehouse

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map

    def name(self):
        return '{}: {}/{}'.format(
            self.__class__.__name__,
            self._database,
            self._warehouse
        )
