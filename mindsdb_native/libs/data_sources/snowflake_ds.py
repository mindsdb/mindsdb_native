import os

from mindsdb_native.libs.data_types.data_source import SQLDataSource


class SnowflakeDS(SQLDataSource):
    def __init__(self, query, host, user, password, account, warehouse,
                 database, schema, protocol='https', port=443):
        super().__init__(query)
        self.host = host
        self.user = user
        self.password = password
        self.account = account
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.protocol = protocol
        self.port = int(port)

    def query(self, q):
        # Note: This import will *break* the requests package in certain cases, guarding against it so that we only touch this odious libray when absolutely necessary (more info here: https://github.com/boto/boto3/issues/2577)
        from snowflake import connector
        con = connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            account=self.account,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema,
            protocol=self.protocol,
            port=self.port
        )
        # Create a cursor object.
        cur = con.cursor()
        cur.execute(q)
        df = cur.fetch_pandas_all()

        cur.close()
        con.close()

        return df, self._make_colmap(df)

    def name(self):
        return 'Snowflake - {}'.format(self._query)
