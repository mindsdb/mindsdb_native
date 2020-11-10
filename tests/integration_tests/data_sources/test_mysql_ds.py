import unittest
from mindsdb_native import F
from . import DB_CREDENTIALS, break_dataset


class TestMYSQL(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['mysql']['user']
        self.PASSWORD = DB_CREDENTIALS['mysql']['password']
        self.HOST = DB_CREDENTIALS['mysql']['host']
        self.PORT = int(DB_CREDENTIALS['mysql']['port'])
        self.DATABASE = 'test_data'
        self.TABLE = 'us_health_insurance'

    def test_mysql_ds(self):
        from mindsdb_native import MySqlDS

        LIMIT = 100

        mysql_ds = MySqlDS(
            table=self.TABLE,
            host=self.HOST,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE,
            port=self.PORT,
            query='SELECT * FROM {} LIMIT {}'.format(self.TABLE, LIMIT)
        )

        mysql_ds.df = break_dataset(mysql_ds.df)

        assert len(mysql_ds) <= LIMIT

        F.analyse_dataset(mysql_ds)
