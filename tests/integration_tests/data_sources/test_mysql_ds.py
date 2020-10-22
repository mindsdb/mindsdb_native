import unittest
import logging
from mindsdb_native import Predictor
from mindsdb_native import F


def test_mysql_ds():
    import mysql.connector
    from mindsdb_native.libs.data_sources.mysql_ds import MySqlDS

    HOST = os.getenv('MYSQL_HOST')
    USER = os.getenv('MYSQL_USER')
    PASSWORD = os.getenv('MYSQL_PASSWORD')
    DATABASE = os.getenv('MYSQL_DATABASE')
    PORT = os.getenv('MYSQL_PORT')

    assert HOST is not None, 'missing environment variable'
    assert USER is not None, 'missing environment variable'
    assert PASSWORD is not None, 'missing environment variable'
    assert DATABASE is not None, 'missing environment variable'
    assert PORT is not None, 'missing environment variable'

    con = mysql.connector.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    cur = con.cursor()

    cur.execute('DROP TABLE IF EXISTS test_mindsdb')
    cur.execute('CREATE TABLE test_mindsdb(col_1 Text, col_2 BIGINT, col_3 BOOL)')
    for i in range(0, 200):
        cur.execute(f'INSERT INTO test_mindsdb VALUES ("This is string number {i}", {i}, {i % 2 == 0})')
    con.commit()
    con.close()

    mysql_ds = MySqlDS(table='test_mindsdb', host=HOST, user=USER,
                       password=PASSWORD, database=DATABASE, port=PORT)

    assert mysql_ds.name() == 'MySqlDS: mysql/test_mindsdb'

    assert (len(mysql_ds._df) == 200)

    mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
    F.analyse_dataset(from_data=mysql_ds)
