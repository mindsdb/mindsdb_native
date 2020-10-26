import os
import unittest
import logging
from mindsdb_native import Predictor
from mindsdb_native import F


class TestMYSQL(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['mysql']['user']
        self.PASSWORD = DB_CREDENTIALS['mysql']['password']
        self.HOST = DB_CREDENTIALS['mysql']['host']
        self.DATABASE = DB_CREDENTIALS['mysql']['database']
        self.PORT = int(DB_CREDENTIALS['mysql']['port'])

    def test_mysql_ds(self):
        import mysql.connector
        from mindsdb_native.libs.data_sources.mysql_ds import MySqlDS

        con = mysql.connector.connect(
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE
        )
        cur = con.cursor()

        cur.execute('DROP TABLE IF EXISTS test_mindsdb')
        cur.execute('CREATE TABLE test_mindsdb(col_1 Text, col_2 BIGINT, col_3 BOOL)')
        for i in range(0, 200):
            cur.execute(f'INSERT INTO test_mindsdb VALUES ("This is string number {i}", {i}, {i % 2 == 0})')
        con.commit()
        con.close()

        mysql_ds = MySqlDS(
            table='test_mindsdb',
            host=self.HOST,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE,
            port=self.PORT
        )

        assert mysql_ds.name() == 'MySqlDS: mysql/test_mindsdb'

        assert (len(mysql_ds._df) == 200)

        mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
        F.analyse_dataset(from_data=mysql_ds)
