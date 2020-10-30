import os
import unittest
from . import DB_CREDENTIALS
import mindsdb_native

class TestSnowflake(unittest.TestCase):
    def test_snowflake_ds(self):
        from mindsdb_native.libs.data_sources.snowflake_ds import SnowflakeDS

        snowflake_ds = mindsdb_native.SnowflakeDS(
            query='SELECT * FROM HEALTHCARE_COSTS',
            host=DB_CREDENTIALS['snowflake']['host'],
            user=DB_CREDENTIALS['snowflake']['user'],
            password=DB_CREDENTIALS['snowflake']['password'],
            account=DB_CREDENTIALS['snowflake']['account'],
            warehouse=DB_CREDENTIALS['snowflake']['warehouse'],
            database=DB_CREDENTIALS['snowflake']['database'],
            schema=DB_CREDENTIALS['snowflake']['schema'],
            protocol=DB_CREDENTIALS['snowflake']['protocol'],
            port=DB_CREDENTIALS['snowflake']['port'],
        )

        predictor = mindsdb_native.Predictor(name='healthcare_cost_predictor')
        predictor.learn(from_data=snowflake_ds, to_predict='CHARGES', stop_training_in_x_seconds=5)
        example_prediction = predictor.predict(when_data={'AGE':24})
        assert 'CHARGES' in example_prediction
