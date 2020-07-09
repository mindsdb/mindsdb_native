import pytest
import requests
from mindsdb_native import Predictor
from mindsdb_native.libs.data_sources.clickhouse_ds import ClickhouseDS
from mindsdb_native import F


@pytest.mark.integration
def test_clickhouse_ds():
    HOST = 'localhost'
    PORT = 8123

    clickhouse_url = f'http://{HOST}:{PORT}'
    queries = [
        'CREATE DATABASE IF NOT EXISTS test',
        'DROP TABLE IF EXISTS test.mock',
        '''
            CREATE TABLE test.mock(
                col1 String
                ,col2 Int64
                ,col3 Array(UInt8)
            ) ENGINE = MergeTree()
                ORDER BY col2
                PARTITION BY col1
        ''',
        "INSERT INTO test.mock VALUES ('a',1,[1,2,3])",
        "INSERT INTO test.mock VALUES ('b',2,[2,3,1])",
        "INSERT INTO test.mock VALUES ('c',3,[3,1,2])"
    ]
    for q in queries:
        r = requests.post(clickhouse_url, data=q)
        assert r.status_code == 200

    clickhouse_ds = ClickhouseDS('SELECT * FROM test.mock ORDER BY col2 DESC LIMIT 2', host=HOST, port=PORT)

    assert (len(clickhouse_ds.df) == 2)
    assert (sum(map(int, clickhouse_ds.df['col2'])) == 5)
    assert (len(list(clickhouse_ds.df['col3'][1])) == 3)
    assert (set(clickhouse_ds.df.columns) == set(['col1', 'col2', 'col3']))

    mdb = Predictor(name='analyse_dataset_test_predictor')
    F.analyse_dataset(from_data=clickhouse_ds)
