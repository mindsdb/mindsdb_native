import pandas as pd
import numpy as np
import pytest

from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.phases.data_cleaner.data_cleaner import DataCleaner


class TestDataCleaner:
    @pytest.fixture()
    def lmd(self, transaction):
        lmd = transaction.lmd
        lmd['empty_columns'] = []
        lmd['columns_to_ignore'] = []
        lmd['predict_columns'] = []
        return lmd

    def test_convert_lists_to_tuples(self, transaction, lmd):

        data_cleaner = DataCleaner(session=transaction.session,
                                     transaction=transaction)

        input_dataframe = pd.DataFrame({
            'lists': [[1, 2, 3] for i in range(10)],
            'arrays': [np.array([1, 2, 3]) for i in range(10)],
            'tuples': [tuple([1, 2, 3]) for i in range(10)],
        })

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        data_cleaner.transaction.input_data = input_data
        data_cleaner.run()

        for col_name in input_data.data_frame:
            for i in range(len(input_dataframe)):
                assert isinstance(input_data.data_frame[col_name].iloc[i], tuple)
