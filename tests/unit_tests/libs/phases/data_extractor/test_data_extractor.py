import random
import unittest

import numpy as np
import pandas as pd

from mindsdb_native import Predictor
from mindsdb_native.libs.controllers.transaction import BreakpointException


class TestDataExtractor(unittest.TestCase):
    def test_apply_f(self):
        predictor = Predictor(name='test_apply_f')
        predictor.breakpoint = 'DataExtractor'

        n_points = 100

        df = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_float': np.linspace(0, n_points, n_points),
            'categorical_str': [f'category_{random.randint(0, 5)}' for i in range(n_points)]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='categorical_str',
                advanced_args={'apply_to_columns': {'categorical_str': lambda x: 'cat'}}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError
        
        assert 1 == len(set(predictor.transaction.input_data.data_frame['categorical_str']))

    def test_inf_values(self):
        predictor = Predictor(name='test_inf_values')
        predictor.breakpoint = 'DataExtractor'

        n_points = 100

        df = pd.DataFrame({
            'numeric_int': [x % 10 for x in list(range(n_points))],
            'numeric_float': np.linspace(0, n_points, n_points)
        })

        df['numeric_int'].iloc[0] = np.inf
        df['numeric_int'].iloc[7] = -np.inf

        df['numeric_float'].iloc[0] = np.inf
        df['numeric_float'].iloc[7] = -np.inf
        df['numeric_float'].iloc[20] = np.inf
        df['numeric_float'].iloc[40] = -np.inf

        try:
            predictor.learn(from_data=df, to_predict='numeric_float')
        except BreakpointException:
            pass
        else:
            raise AssertionError
        
        null_count = 0
        for col in predictor.transaction.input_data.data_frame.columns:
            null_count += predictor.transaction.input_data.data_frame[col].isna().sum()

        assert null_count == 6
