from integration_tests.data_sources import ClickhouseTest, break_dataset
from mindsdb_native import F


class TestBrokenDatasets(ClickhouseTest):
    def get_ds(self, table, limit=300):
        from mindsdb_native import ClickhouseDS
        return ClickhouseDS(
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD,
            query='SELECT * FROM {}.{} LIMIT {}'.format(
                self.DATABASE,
                table,
                limit
            )
        )

    def test_home_rentals(self):
        ds = self.get_ds('home_rentals', limit=500)
        stats_1 = F.analyse_dataset(ds)['data_analysis_v2']

        ds.df = break_dataset(ds.df)
        stats_2 = F.analyse_dataset(ds)['data_analysis_v2']

        for col in ds.df.columns:
            if col in stats_1 and col in stats_2:
                assert stats_1[col]['typing']['data_type'] == stats_2[col]['typing']['data_type']
            else:
                if not (col not in stats_1 and col not in stats_2):
                    raise AssertionError

    def test_hdi(self):
        ds = self.get_ds('hdi', limit=500)
        stats_1 = F.analyse_dataset(ds)['data_analysis_v2']

        ds.df = break_dataset(ds.df)
        stats_2 = F.analyse_dataset(ds)['data_analysis_v2']

        for col in ds.df.columns:
            if col in stats_1 and col in stats_2:
                assert stats_1[col]['typing']['data_type'] == stats_2[col]['typing']['data_type']
            else:
                if not (col not in stats_1 and col not in stats_2):
                    raise AssertionError

    def test_us_health_insurance(self):
        ds = self.get_ds('us_health_insurance', limit=500)
        stats_1 = F.analyse_dataset(ds)['data_analysis_v2']

        ds.df = break_dataset(ds.df)
        stats_2 = F.analyse_dataset(ds)['data_analysis_v2']

        for col in ds.df.columns:
            if col in stats_1 and col in stats_2:
                assert stats_1[col]['typing']['data_type'] == stats_2[col]['typing']['data_type']
            else:
                if not (col not in stats_1 and col not in stats_2):
                    raise AssertionError
