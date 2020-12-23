from pyathena import connect
from pyathena.pandas.util import as_pandas

from mindsdb_native.libs.data_types.data_source import SQLDataSource


class AthenaDS(SQLDataSource):
    def __init__(self, query, staging_dir, database=None,
                 access_key=None, secret_key=None, region_name=None, table=None):
        """
        :param query: Query to be executed. Ex. SELECT * FROM db.table;
        :param staging_dir: Full S3 path where Athena temp data will stored. Ex. s3://bucket_name/athena/staging
        :param database: Name of the Database
        :param table: Name of the Table
        :param access_key: Access Key used if supplied else used default credentials.
        :param secret_key: Secret Key used if supplied else used default credentials.
        :param region_name: Region used if supplied else used default region.
        """
        super().__init__(query)

        if (not database or not table) and not query:
            raise ValueError('Either database and table or query should be passed.')
        
        self.staging_dir = staging_dir
        self.database = database
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name

    def query(self, q):
        _conn_args = {
            's3_staging_dir': self.staging_dir,
            'database': self.database
        }

        if self.access_key is not None and self.secret_key is not None:
            _conn_args['aws_access_key_id'] = self.access_key
            _conn_args['aws_secret_access_key'] = self.secret_key

        if self.region_name is not None:
            _conn_args['region_name'] = self.region_name

        conn = connect(**_conn_args)
        cursor = conn.cursor()

        cursor.execute(q)

        # Load query results into Pandas DataFrame and show results
        df = as_pandas(cursor)

        col_map = {}
        for col in df.columns:
            col_map[col] = col

        return df, col_map

    def name(self):
        return 'Athena - {}'.format(self._query)
