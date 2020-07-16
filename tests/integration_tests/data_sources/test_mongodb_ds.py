import pytest
import logging
from pymongo import MongoClient

from mindsdb_native import Predictor
from mindsdb_native.libs.data_sources.mongodb_ds import MongoDS
from mindsdb_native import F


@pytest.mark.integration
def test_mongodb_ds():

    HOST = 'localhost'
    USER = 'root'
    PASSWORD = '123'
    DATABASE = 'database'
    COLLECTION_NAME = 'test_mindsdb'
    PORT = 27017

    con = MongoClient(host=HOST,
                      port=PORT,
                      username=USER,
                      password=PASSWORD)

    db = con[DATABASE]
    
    if COLLECTION_NAME in db.list_collection_names():
        db[COLLECTION_NAME].drop()

    collection = db[COLLECTION_NAME]

    for i in range(0, 200):
        collection.insert_one({
            'col_1': "This is string number {}".format(i),
            'col_2': i,
            'col_3': (i % 2) == 0
        })

    mongodb_ds = MongoDS(collection=COLLECTION_NAME,
                         query={},
                         host=HOST,
                         port=PORT,
                         user=USER,
                         password=PASSWORD,
                         database=DATABASE)

    assert mongodb_ds.name() == 'MongoDS: database/test_mindsdb'

    assert (len(mongodb_ds._df) == 200)

    mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
    F.analyse_dataset(from_data=mongodb_ds)
