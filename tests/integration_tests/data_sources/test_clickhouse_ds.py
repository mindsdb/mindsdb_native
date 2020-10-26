import os
import json
import unittest
import requests
import mindsdb_native
from mindsdb_native import Predictor
from mindsdb_native import F
from . import DB_CREDENTIALS


class TestClickhouse(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['clickhouse']['user']
        self.PASSWORD = DB_CREDENTIALS['clickhouse']['password']
        self.HOST = DB_CREDENTIALS['clickhouse']['host']
        self.PORT = int(DB_CREDENTIALS['clickhouse']['port'])

    def test_clickhouse_ds(self):
        from mindsdb_native.libs.data_sources.clickhouse_ds import ClickhouseDS

        clickhouse_url = f'http://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}'

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

        clickhouse_ds = ClickhouseDS(
            'SELECT * FROM test.mock ORDER BY col2 DESC LIMIT 2',
            host=self.HOST,
            port=self.PORT
        )

        assert (len(clickhouse_ds.df) == 2)
        assert (sum(map(int, clickhouse_ds.df['col2'])) == 5)
        assert (len(list(clickhouse_ds.df['col3'][1])) == 3)
        assert (set(clickhouse_ds.df.columns) == set(['col1', 'col2', 'col3']))

        mdb = Predictor(name='analyse_dataset_test_predictor')
        F.analyse_dataset(from_data=clickhouse_ds)

    def test_database_history(self):
        from mindsdb_native.libs.data_sources.clickhouse_ds import ClickhouseDS

        clickhouse_url = f'http://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}'

        values = []
        for i in range(500):
            values.append([str(i%4),i,i*2])

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

        clickhouse_ds = ClickhouseDS(
            'SELECT * FROM test.mock',
            host=self.HOST,
            port=self.PORT
        )

        mindsdb_native.Predictor(name='query_history_based_ts_predictor').learn(
            to_predict='col3',
            from_data=clickhouse_ds,
            stop_training_in_x_seconds=5,
            timeseries_settings={
                'order_by': ['col2']
                ,'window': 6
                ,'group_by': ['col1']
            }
        )

        ts_predictor = mindsdb_native.Predictor(name='query_history_based_ts_predictor')
        ts_predictor.predict(when_data={
            'col2': 800
            ,'col1': '2'
        })
