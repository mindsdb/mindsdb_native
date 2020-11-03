import unittest
from mindsdb_native import F
from . import DB_CREDENTIALS, break_dataset


class TestMariaDB(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['mariadb']['user']
        self.PASSWORD = DB_CREDENTIALS['mariadb']['password']
        self.HOST = DB_CREDENTIALS['mariadb']['host']
        self.PORT = int(DB_CREDENTIALS['mariadb']['port'])
        self.DATABASE = 'test_data'
        self.TABLE = 'hdi'

    def test_maria_ds(self):
        from mindsdb_native import MariaDS

        LIMIT = 100

        maria_ds = MariaDS(
            table=self.TABLE,
            host=self.HOST,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE,
            port=self.PORT,
            query='SELECT * FROM {} LIMIT {}'.format(self.TABLE, LIMIT)
        )

        maria_ds.df = break_dataset(maria_ds.df)

        assert len(maria_ds) <= LIMIT

        F.analyse_dataset(from_data=maria_ds)
