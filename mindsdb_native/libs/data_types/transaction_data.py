

class TransactionData:
    def __init__(self):
        self.data_frame = None
        self.train_df = None
        self.test_df = None
        self.validation_df = None

        self._sample_df = None

    @property
    def columns(self):
        if self.data_frame is not None:
            return self.data_frame.columns
        else:
            return self.train_df.columns

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
