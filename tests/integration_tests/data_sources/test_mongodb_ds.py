import os
import unittest
import logging
from mindsdb_native import Predictor
from mindsdb_native import F
from . import DB_CREDENTIALS
from . import DB_CREDENTIALS, RUN_ID


class TestMongoDB(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['mongodb']['user']
        self.PASSWORD = DB_CREDENTIALS['mongodb']['password']
        self.HOST = DB_CREDENTIALS['mongodb']['host']
        self.PORT = int(DB_CREDENTIALS['mongodb']['port'])
        self.DATABASE = 'test_db'
        self.COLLECTION = 'test_collection'
        if RUN_ID is not None:
            self.DATABASE += '_' + RUN_ID
            self.COLLECTION += '_' + RUN_ID

    # @unittest.skip('pymongo.errors.ServerSelectionTimeoutError')
    def test_mongodb_ds(self):
        from pymongo import MongoClient
        from mindsdb_native.libs.data_sources.mongodb_ds import MongoDS

        con = MongoClient(
            host=self.HOST,
            port=int(self.PORT),
            username=self.USER,
            password=self.PASSWORD
        )

        db = con[self.DATABASE]
        
        if self.COLLECTION in db.list_collections():
            db[self.COLLECTION].drop()

        collection = db[self.COLLECTION]

        for i in range(0, 200):
            collection.insert_one({
                'col_1': "This is string number {}".format(i),
                'col_2': i,
                'col_3': (i % 2) == 0
            })

        mongodb_ds = MongoDS(
            collection=self.COLLECTION,
            query={},
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE
        )

        assert (len(mongodb_ds._df) == 200)

        mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
        F.analyse_dataset(from_data=mongodb_ds)
