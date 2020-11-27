import time
import random
import string
import unittest
import requests
from . import DB_CREDENTIALS, break_dataset, ClickhouseTest
from mindsdb_native import Predictor, F
import gc


def random_string():
    return ''.join(
        random.Random(
            int(time.time() * 1e6)
        ).choices(
            string.ascii_letters + string.digits,
            k=6
        )
    )


class TestClickhouse(ClickhouseTest):
    def test_clickhouse_ds(self):
        from mindsdb_native import ClickhouseDS
        LIMIT = 100

        clickhouse_ds = ClickhouseDS(
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD,
            query='SELECT * FROM {}.{} LIMIT {}'.format(
                self.DATABASE,
                'home_rentals',
                LIMIT
            )
        )

        # test filter
        for val in clickhouse_ds.filter([['location', 'like','ood']])['location']:
            assert val == 'good'

        assert len(clickhouse_ds.filter([['rental_price', '>', 2500]], 3)) == 3
        assert len(clickhouse_ds.filter([['initial_price', '<', 0]], 3)) == 0

        # mess with the values inside then try to analyze it
        clickhouse_ds.df = break_dataset(clickhouse_ds.df)
        assert len(clickhouse_ds) <= LIMIT
        F.analyse_dataset(from_data=clickhouse_ds)

    def test_database_history(self):
        return
        from mindsdb_native import ClickhouseDS

        TEMP_DB = 'test_database_history_' + random_string()
        TEMP_TABLE = 'tmp_test_database_history_' + random_string()

        params = {'user': self.USER, 'password': self.PASSWORD}

        clickhouse_url = f'http://{self.HOST}:{self.PORT}'

        values = []
        for i in range(200):
            values.append([str(i % 4), i, i * 2])

        queries = [
            f'CREATE DATABASE IF NOT EXISTS {TEMP_DB}',
            f'DROP TABLE IF EXISTS {TEMP_DB}.{TEMP_TABLE}',
            f'''
                CREATE TABLE {TEMP_DB}.{TEMP_TABLE}(
                    col1 String
                    ,col2 Int64
                    ,col3 Int64
                ) ENGINE = MergeTree()
                    ORDER BY col2
                    PARTITION BY col1
            ''',
        ]
        gc.collect()

        for value in values:
            value_ins_str = str(value).replace('[','').replace(']','')
            queries.append(f"INSERT INTO {TEMP_DB}.{TEMP_TABLE} VALUES ({value_ins_str})")

        for q in queries:
            r = requests.post(clickhouse_url, data=q, params=params)
            assert r.status_code == 200, r.text

        clickhouse_ds = ClickhouseDS(
            f'SELECT * FROM {TEMP_DB}.{TEMP_TABLE}',
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD
        )

        temp_predictor = Predictor(name='query_history_based_ts_predictor')
        temp_predictor.learn(
            to_predict='col3',
            from_data=clickhouse_ds,
            stop_training_in_x_seconds=5,
            timeseries_settings={
                'order_by': ['col2']
                ,'window': 6
                ,'group_by': ['col1']
            }
        )
        del temp_predictor

        ts_predictor = Predictor(name='query_history_based_ts_predictor')
        predictions = ts_predictor.predict(
            when_data={
                'col2': 800
                ,'col1': '2'
            },
            advanced_args={
                'use_database_history': True
            }
        )

        assert predictions[0]['col3'] is not None

        r = requests.post(
            clickhouse_url,
            data=f'DROP DATABASE {TEMP_DB}',
            params=params
        )
        assert r.status_code == 200, 'failed to drop temporary database "{}"'.format(TEMP_DB)
