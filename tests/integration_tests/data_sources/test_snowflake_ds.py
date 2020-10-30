import os
import unittest
from . import DB_CREDENTIALS
import mindsdb_native

class TestSnowflake(unittest.TestCase):
    def test_snowflake_ds(self):
        from mindsdb_native.libs.data_sources.snowflake_ds import SnowflakeDS

        # Create the datasource
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

        # Make sure we can use it for some basic tasks
        data_analysis = mindsdb_native.F.analyse_dataset(snowflake_ds, sample_settings={
            'sample_percentage': 5
        })
        assert len(data_analysis['data_analysis_v2']['columns']) == 7
