import pandas as pd
import numpy as np
import random
import pytest

from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.phases.data_splitter.data_splitter import DataSplitter
from mindsdb_native.libs.constants.mindsdb import TRANSACTION_LEARN


class TestDataSplitter:
    @pytest.fixture()
    def lmd(self, transaction):
        lmd = transaction.lmd
        lmd['tss'] = {}
        lmd['tss']['order_by'] = 'ob'
        lmd['tss']['group_by'] = ['gb_1', 'gb_2']
        lmd['type'] = TRANSACTION_LEARN
        lmd['data_preparation'] = {}
        return lmd

    def test_groups(self, transaction, lmd):
        data_splitter = DataSplitter(
            session=transaction.session,
            transaction=transaction
        )

        ob = [*range(100)]
        random.shuffle(ob)

        input_dataframe = pd.DataFrame({
            'ob': ob,
            'gb_1': [1, 1, 2, 2] * 25,
            'gb_2': [1, 2, 1, 2] * 25
        })

        input_data = TransactionData()
        input_data.data_frame = input_dataframe

        data_splitter.transaction.input_data = input_data
        all_indexes, *_ = data_splitter.run(test_train_ratio=0.25)

        assert len(all_indexes[(1, 1)]) == 25
        assert len(all_indexes[(1, 2)]) == 25
        assert len(all_indexes[(2, 1)]) == 25
        assert len(all_indexes[(2, 2)]) == 25
        assert len(all_indexes[tuple()]) == 100
