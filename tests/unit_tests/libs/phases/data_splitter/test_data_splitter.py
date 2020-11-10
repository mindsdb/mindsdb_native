import pandas as pd
import random

from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException


class TestDataSplitter:
    def test_user_provided_split_indices(self):
        predictor = Predictor('test_user_provided_split_indices')

        df = pd.DataFrame({
            'col_a': [*range(100)],
            'col_b': [*range(100)]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='col_b',
                advanced_args={
                    'data_split_indexes': {
                        'validation_indexes': [*range(0, 30)],
                        'train_indexes': [*range(30, 60)],
                        'test_indexes': [*range(60, 100)]
                    },
                    'force_column_usage': ['col_a', 'col_b']
                },
                use_gpu=False
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert set(predictor.transaction.input_data.train_df['col_a'].tolist()) == set(range(30, 60))
        assert set(predictor.transaction.input_data.test_df['col_a'].tolist()) == set(range(60, 100))
        assert set(predictor.transaction.input_data.validation_df['col_a'].tolist()) == set(range(0, 30))

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
