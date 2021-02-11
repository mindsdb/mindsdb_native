import os
import unittest
import mindsdb_native
from . import DB_CREDENTIALS, break_dataset


class TestSnowflake(unittest.TestCase):
    @unittest.skip('Snowflake too buggy')
    def test_snowflake_ds(self):
        if os.name == 'nt':
            print('Snowflake datasource (SnowflakeDS) can\'t be used on windows at the moment due to the connector not working')
            return

        from mindsdb_native import SnowflakeDS

        # Create the datasource
        snowflake_ds = SnowflakeDS(
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

        snowflake_ds.df = break_dataset(snowflake_ds.df)

        # Make sure we can use it for some basic tasks
        data_analysis = mindsdb_native.F.analyse_dataset(
            snowflake_ds,
            sample_settings={'sample_percentage': 5}
        )

        assert len(data_analysis['columns']) == 7
