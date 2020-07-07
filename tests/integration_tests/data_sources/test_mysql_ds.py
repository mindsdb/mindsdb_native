import pytest
import mysql.connector
import logging
from mindsdb_native import Predictor
from mindsdb_native.libs.data_sources.mysql_ds import MySqlDS
from mindsdb_native import F

@pytest.mark.integration
def test_mysql_ds():
    HOST = 'localhost'
    USER = 'root'
    PASSWORD = ''
    DATABASE = 'mysql'
    PORT = 3306

    con = mysql.connector.connect(host=HOST,
                                  port=PORT,
                                  user=USER,
                                  password=PASSWORD,
                                  database=DATABASE)
    cur = con.cursor()

    cur.execute('DROP TABLE IF EXISTS test_mindsdb')
    cur.execute('CREATE TABLE test_mindsdb(col_1 Text, col_2 BIGINT, col_3 BOOL)')
    for i in range(0, 200):
        cur.execute(f'INSERT INTO test_mindsdb VALUES ("This is string number {i}", {i}, {i % 2 == 0})')
    con.commit()
    con.close()

    mysql_ds = MySqlDS(table='test_mindsdb', host=HOST, user=USER,
                       password=PASSWORD, database=DATABASE, port=PORT)
    assert (len(mysql_ds._df) == 200)

    mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
    F.analyse_dataset(from_data=mysql_ds)
