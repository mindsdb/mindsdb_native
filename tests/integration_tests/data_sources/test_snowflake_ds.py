import os
import unittest
from . import DB_CREDENTIALS


class TestSnowflake(unittest.TestCase):
    @unittest.skip(reason='Can\'t run snowflake in a docker container')
    def test_snowflake_ds(self):
        from mindsdb_native.libs.data_sources.snowflake_ds import SnowflakeDS

        # Create a snowflake datasource
        snowflake_ds = SnowflakeDS(
            query='SELECT * FROM HEALTHCARE_COSTS',
            host='zka81761.us-east-1.snowflakecomputing.com',
            user='GEORGE3D6',
            password='',
            account='zka81761.us-east-1.aws',
            warehouse='COMPUTE_WH',
            database='DEMO_DB',
            schema='PUBLIC',
            protocol='https',
            port=443
        )

        '''
        The schema for `HEALTHCARE_COSTS` is:
        AGE         NUMBER
        SEX         STRING
        BMI         FLOAT
        CHILDREN    STRING
        SMOKER      STRING
        REGION      STRING
        CHARGES     FLOAT
        '''

        # We want to ask mindsdb with predicting the hosptialization charges given we have the rest of the information about the patient: Age, Sex, BMI, Nr. Children, Smoker status and region

        import mindsdb
        predictor = mindsdb.Predictor(name='healthcare_cost_predictor')
        predictor.learn(from_data=snowflake_ds, to_predict='CHARGES')
        example_prediction = predictor.predict(when_data={'AGE':24})
        print(example_prediction)
