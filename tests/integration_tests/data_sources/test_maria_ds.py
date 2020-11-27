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

        LIMIT = 200

        maria_ds = MariaDS(
            host=self.HOST,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE,
            port=self.PORT,
            query='SELECT * FROM `{}` LIMIT {}'.format(self.TABLE, LIMIT)
        )

        maria_ds.df = break_dataset(maria_ds.df)

        assert len(maria_ds) <= LIMIT

        F.analyse_dataset(from_data=maria_ds)

        # Our SQL parsing succeds here, but the query fails, test if we're still able to filter via the dataframe fallback
        maria_ds._query = maria_ds._query.replace(self.TABLE, 'wrongly_named_table')
        assert len(maria_ds.filter([['Population', '<', 33098932]], 8)) == 8
        assert len(maria_ds.filter([['Development_Index', '!=', 3]], 12)) == 12
