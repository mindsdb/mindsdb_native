import os
import unittest
import datetime
import logging
from mindsdb_native import Predictor
from mindsdb_native import F


class TestPostgres(unittest.TestCase):
    def test_postgres_ds(self):
        import pg8000
        from mindsdb_native.libs.data_sources.postgres_ds import PostgresDS

        HOST = os.getenv('POSTGRES_HOST')
        USER = os.getenv('POSTGRES_USER')
        PASSWORD = os.getenv('POSTGRES_PASSWORD')
        DBNAME = os.getenv('POSTGRES_DBNAME')
        PORT = os.getenv('POSTGRES_PORT')

        assert HOST is not None, 'missing environment variable'
        assert USER is not None, 'missing environment variable'
        assert PASSWORD is not None, 'missing environment variable'
        assert DBNAME is not None, 'missing environment variable'
        assert PORT is not None, 'missing environment variable'

        PORT = int(PORT)

        con = pg8000.connect(
            database=DBNAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        cur = con.cursor()

        cur.execute('DROP TABLE IF EXISTS test_mindsdb')
        cur.execute(
            'CREATE TABLE test_mindsdb(col_1 Text, col_2 Int,  col_3 Boolean, col_4 Date, col_5 Int [])')
        for i in range(0, 200):
            dt = datetime.datetime.now() - datetime.timedelta(days=i)
            dt_str = dt.strftime('%Y-%m-%d')
            cur.execute(
                f'INSERT INTO test_mindsdb VALUES (\'String {i}\', {i}, {i % 2 == 0}, \'{dt_str}\', ARRAY [1, 2, {i}])')
        con.commit()
        con.close()

        postgres_ds = PostgresDS(table='test_mindsdb', host=HOST, user=USER,
                            password=PASSWORD, database=DBNAME, port=PORT)
                            
        assert postgres_ds.name() == 'PostgresDS: postgres/test_mindsdb'

        assert (len(postgres_ds._df) == 200)

        mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
        F.analyse_dataset(from_data=postgres_ds)
