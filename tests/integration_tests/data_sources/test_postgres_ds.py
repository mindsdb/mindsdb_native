import unittest
from mindsdb_native import F
from . import DB_CREDENTIALS, break_dataset


class TestPostgres(unittest.TestCase):
    def setUp(self):
        self.USER = DB_CREDENTIALS['postgres']['user']
        self.PASSWORD = DB_CREDENTIALS['postgres']['password']
        self.HOST = DB_CREDENTIALS['postgres']['host']
        self.PORT = int(DB_CREDENTIALS['postgres']['port'])
        self.DATABASE = 'postgres'
        self.TABLE = 'home_rentals'

    def test_postgres_ds(self):
        from mindsdb_native import PostgresDS

        LIMIT = 100

        postgres_ds = PostgresDS(
            table=self.TABLE,
            host=self.HOST,
            user=self.USER,
            password=self.PASSWORD,
            database=self.DATABASE,
            port=self.PORT,
            query='SELECT * FROM {}.{} LIMIT {}'.format(
                'test_data',
                self.TABLE,
                LIMIT
            )
        )

        postgres_ds.df = break_dataset(postgres_ds.df)
                            
        assert len(postgres_ds) == LIMIT

        F.analyse_dataset(postgres_ds)
