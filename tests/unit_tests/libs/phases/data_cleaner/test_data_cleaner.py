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

    def test_ignore_columns(self, transaction, lmd):
        data_cleaner = DataCleaner(session=transaction.session,
                                   transaction=transaction)

        input_dataframe = pd.DataFrame({
            'do_use': [1, 2, 3],
            'ignore_this': [0, 1, 100]
        })

        lmd['columns_to_ignore'].append('ignore_this')

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        data_cleaner.transaction.input_data = input_data
        data_cleaner.run()

        assert 'do_use' in input_data.data_frame.columns
        assert 'ignore_this' not in input_data.data_frame.columns

