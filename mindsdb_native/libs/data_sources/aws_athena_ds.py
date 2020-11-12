from pyathena import connect
from pyathena.util import as_pandas

from mindsdb_native.libs.data_types.data_source import DataSource


class AthenaDS(DataSource):
    def __init__(self, *args, **kwargs):
        self.is_sql = True
        super(AthenaDS, self).__init__(*args, **kwargs)

    def _setup(self, staging_dir, database=None, table=None, query=None,
               access_key=None, secret_key=None, region_name=None):
        """

        :param staging_dir: Full S3 path where Athena temp data will stored. Ex. s3://bucket_name/athena/staging
        :param database: Name of the Database
        :param table: Name of the Table
        :param query: Query to be executed. Ex. SELECT * FROM db.table;
        :param access_key: Access Key used if supplied else used default credentials.
        :param secret_key: Secret Key used if supplied else used default credentials.
        :param region_name: Region used if supplied else used default region.
        """

        if (not database or not table) and not query:
            raise ValueError("Either database and table or query should be passed.")

        self._database_name = database
        self._table_name = table

        _conn_args = {
            's3_staging_dir': staging_dir,
            'database': database,
            'table': table
        }

        if access_key is not None and secret_key is not None:
            _conn_args['aws_access_key_id'] = access_key
            _conn_args['aws_secret_access_key'] = secret_key

        if region_name:
            _conn_args['region_name'] = region_name

        conn = connect(**_conn_args)
        cursor = conn.cursor()

        if query:
            cursor.execute(query)
        else:
            cursor.execute("SELECT * FROM {database}.{table};".format(database=database,
                                                                      table=table))

        # Load query results into Pandas DataFrame and show results
        df = as_pandas(cursor)

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
