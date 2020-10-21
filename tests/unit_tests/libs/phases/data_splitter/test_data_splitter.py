import pandas as pd
import numpy as np
import random
import pytest

from mindsdb_native.config import CONFIG
from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException
from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.phases.data_splitter.data_splitter import DataSplitter
from mindsdb_native.libs.constants.mindsdb import TRANSACTION_LEARN


class TestDataSplitter:
    def test_groups(self):
        predictor = Predictor('test_groups')
        predictor.breakpoint = 'DataSplitter'

        ob = [*range(100)]
        random.shuffle(ob)

        df = pd.DataFrame({
            'ob': ob,
            'gb_1': [1, 1, 2, 2] * 25,
            'gb_2': [1, 2, 1, 2] * 25
        })

        try:
            predictor.learn(from_data=df, to_predict='ob')
        except BreakpointException as e:
            all_indexes, *_ = e.ret
        else:
            raise AssertionError

        assert len(all_indexes[(1, 1)]) == 25
        assert len(all_indexes[(1, 2)]) == 25
        assert len(all_indexes[(2, 1)]) == 25
        assert len(all_indexes[(2, 2)]) == 25
        assert len(all_indexes[tuple()]) == 100
