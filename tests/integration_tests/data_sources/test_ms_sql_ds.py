import pytest
import logging
from mindsdb_native import Predictor
from mindsdb_native import F


@pytest.mark.integration
def test_mssql_ds():
    import pytds
    from mindsdb_native.libs.data_sources.ms_sql_ds import MSSQLDS

    HOST = 'localhost'
    USER = 'sa'
    PASSWORD = '123ABCdef@!'
    DATABASE = 'master'
    PORT = 1433

    with pytds.connect(dsn=HOST,
                       user=USER,
                       password=PASSWORD,
                       database=DATABASE) as con:
        with con.cursor() as cur:
            cur.execute("IF OBJECT_ID('dbo.test_mindsdb') IS NOT NULL DROP TABLE dbo.test_mindsdb")
            cur.execute('CREATE TABLE test_mindsdb(col_1 Text, col_2 BIGINT, col_3 BIT)')
            for i in range(0, 200):
                cur.execute(f"INSERT INTO test_mindsdb ([col_1], [col_2], [col_3]) VALUES ('This is string number {i}', {i}, {i % 2})")
        con.commit()

    mssql_ds = MSSQLDS(table='test_mindsdb', host=HOST, user=USER,
                       password=PASSWORD, database=DATABASE, port=PORT)

    assert mssql_ds.name() == 'MSSQLDS: master/test_mindsdb'

    assert (len(mssql_ds._df) == 200)

    mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
    F.analyse_dataset(from_data=mssql_ds)
