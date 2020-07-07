from mindsdb_native.libs.helpers.stats_helpers import sample_data


class TransactionData:
    def __init__(self):
        self.data_frame = None
        self.columns = []

        self._sample_df = None

        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None

    @property
    def train_df(self):
        return self.data_frame.loc[self.train_idx]

    @property
    def test_df(self):
        return self.data_frame.loc[self.test_idx]

    @property
    def validation_df(self):
        return self.data_frame.loc[self.validation_idx]

    def sample_df(self,
                  sample_function,
                  sample_margin_of_error,
                  sample_confidence_level,
                  sample_percentage):
        if self._sample_df is None:
            self._sample_df = sample_function(self.data_frame,
                                          sample_margin_of_error,
                                          sample_confidence_level,
                                          sample_percentage)

        return self._sample_df
