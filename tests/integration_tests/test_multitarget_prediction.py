from integration_tests.data_sources import ClickhouseTest, break_dataset
from mindsdb_native import Predictor, ClickhouseDS


class TestMultitargetPrediction(ClickhouseTest):
    def test_multitarget_prediction(self):
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

        clickhouse_ds.df = break_dataset(clickhouse_ds.df)

        assert len(clickhouse_ds) <= LIMIT

        p = Predictor('test_multitarget_prediction')

        p.learn(
            from_data=clickhouse_ds,
            to_predict=['rental_price', 'location'],
            stop_training_in_x_seconds=3,
            use_gpu=False,
            advanced_args={'debug': True}
        )