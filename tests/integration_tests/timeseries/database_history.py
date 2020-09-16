import pytest
import mindsdb_native

@pytest.mark.integration
def test_database_history():
    from mindsdb_native.libs.data_sources.clickhouse_ds import ClickhouseDS

    HOST = 'localhost'
    PORT = 8123

    clickhouse_url = f'http://{HOST}:{PORT}'

    values = []
    for i in range(200):
        values.append([str(i%6),i,i*2])

    queries = [
        'CREATE DATABASE IF NOT EXISTS test',
        'DROP TABLE IF EXISTS test.mock',
        '''
            CREATE TABLE test.mock(
                col1 String
                ,col2 Int64
                ,col3 Int64
            ) ENGINE = MergeTree()
                ORDER BY col2
                PARTITION BY col1
        ''',
    ]

    for value in values:
        value_ins_str = str(value).replace('[','').replace(']','')
        queries.append(f"INSERT INTO test.mock VALUES ({value_ins_str})")

    for q in queries:
        r = requests.post(clickhouse_url, data=q)
        assert r.status_code == 200

    clickhouse_ds = ClickhouseDS('SELECT * FROM test.mock', host=HOST, port=PORT)

    ts_predictor = mindsdb_native.Predictor(name='query_history_based_ts_predictor')
