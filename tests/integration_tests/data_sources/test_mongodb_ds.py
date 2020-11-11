import unittest
from mindsdb_native import Predictor, F
from . import DB_CREDENTIALS


class TestMongoDB(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['mongodb']['user']
        self.PASSWORD = DB_CREDENTIALS['mongodb']['password']
        self.HOST = DB_CREDENTIALS['mongodb']['host']
        self.PORT = int(DB_CREDENTIALS['mongodb']['port'])
        self.DATABASE = 'test_data'
        self.COLLECTION = 'home_rentals'

        # test filter (this is a non sql ds)
        assert not clickhouse_ds.is_sql
        print(clickhouse_ds.filter([['location', 'like','oo']])['location'])
        for val in clickhouse_ds.filter([['location', 'like','oo']])['location']:
            assert val == 'good'

        assert len(clickhouse_ds.filter([['rental_price', '>', 2500]], 3)) == 3
        assert len(clickhouse_ds.filter([['initial_price', '<', 0]], 3)) == 0


    def test_mongodb_ds(self):
        from mindsdb_native import MongoDS

        mongodb_ds = MongoDS(
            collection=self.COLLECTION,
            query={},
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE
        )

        F.analyse_dataset(from_data=mongodb_ds)
