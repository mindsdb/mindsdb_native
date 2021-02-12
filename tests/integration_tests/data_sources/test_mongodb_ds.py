import unittest
from mindsdb_native import Predictor, F
from . import DB_CREDENTIALS
from mindsdb_native import MongoDS


class TestMongoDB(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['mongodb']['user']
        self.PASSWORD = DB_CREDENTIALS['mongodb']['password']
        self.HOST = DB_CREDENTIALS['mongodb']['host']
        self.PORT = int(DB_CREDENTIALS['mongodb']['port'])
        self.DATABASE = 'test_data'
        self.COLLECTION = 'home_rentals'

    def test_mongodb_ds(self):

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

        for val in mongodb_ds.filter([['location', 'like','ood']])['location']:
            assert val == 'good'

        assert len(mongodb_ds.filter([['rental_price', '>', 2500]], 3)) == 3
        assert len(mongodb_ds.filter([['initial_price', '<', 0]], 3)) == 0
    
    def test_mongodb_atlas_ds(self):
        USER = DB_CREDENTIALS['mongodb_atlas']['user']
        PASSWORD = DB_CREDENTIALS['mongodb_atlas']['password']
        HOST = DB_CREDENTIALS['mongodb_atlas']['host']
        PORT = int(DB_CREDENTIALS['mongodb_atlas']['port'])
        DATABASE = 'sample_restaurants'
        COLLECTION = 'restaurants'
        atlas_ds = MongoDS(
            collection=COLLECTION,
            query={"grades": []},
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE,
            limit=5,
                )
        assert len(atlas_ds.df) == 5

        df, _ = atlas_ds.query({"grades":{"$size": 5}})
        assert len(df) == 5
        df, _ = atlas_ds.query({"grades":{"$size": 45}})
        assert len(df) == 0
