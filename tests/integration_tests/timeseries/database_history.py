import pytest
import requests
import mindsdb_native

@pytest.mark.integration
def test_database_history():
    from mindsdb_native.libs.data_sources.clickhouse_ds import ClickhouseDS

    HOST = 'localhost'
    PORT = 8123

    clickhouse_url = f'http://{HOST}:{PORT}'

    values = []
    for i in range(200):
        values.append([str(i%6),i,pow(i,2)])

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

    mindsdb_native.Predictor(name='query_history_based_ts_predictor').learn(to_predict='col3', from_data=clickhouse_ds, timeseries_settings={
        'order_by': ['col2']
        ,'window': 6
        ,'group_by': ['col1']
    }, stop_training_in_x_seconds=5)


    ts_predictor = mindsdb_native.Predictor(name='query_history_based_ts_predictor')
    ts_predictor.predict(when_data={
        'col2': 200
        ,'col1': '2'
    })
