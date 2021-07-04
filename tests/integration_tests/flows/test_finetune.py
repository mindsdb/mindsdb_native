import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, balanced_accuracy_score
np.random.seed(42)

import mindsdb_native
from mindsdb_native import Predictor


class TestFinetune(unittest.TestCase):
    def finetune_helper(self, df, target, grouped_col, acc_func, pred_name):
        test_size=0.2
        groups = df[grouped_col].unique()
        data_tuples = []

        for group in groups:
            grouped = df[df[grouped_col] == group]
            train = grouped[:int(len(grouped)*(1-test_size))]
            test = grouped[int(len(grouped)*(1-test_size)):]
            if len(data_tuples) > 0:
                all_test = test.append(data_tuples[-1][1])  # adds previous groups to the latest test DF
            else:
                all_test = test
            data_tuples.append((train, test, all_test))

        predictor = Predictor(name=pred_name)
        results_single_test = []
        results_all_test = []
        for i, (train, test, all_test) in enumerate(data_tuples):
            if i == 0:
                predictor.learn(from_data=train,
                                to_predict=target,
                                stop_training_in_x_seconds=10,
                                use_gpu=False)
            else:
                # finetune here
                predictor.adjust(from_data=train)

            result_single = predictor.test(when_data=test, accuracy_score_functions=acc_func)
            result_all = predictor.test(when_data=all_test, accuracy_score_functions=acc_func)

            results_single_test.append(result_single[f'{target}_accuracy'])
            results_all_test.append(result_all[f'{target}_accuracy'])

        return results_single_test, results_all_test

    def test_finetune_numerical(self):
        df = pd.read_csv("https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv")
        accs_single_split, accs_all_data = self.finetune_helper(df,
                                                                'rental_price',
                                                                'neighborhood',
                                                                r2_score,
                                                                '__test_finetune_numerical')
        assert np.array(accs_single_split).mean() >= 0.8
        assert np.array(accs_all_data).mean() >= 0.8

    def test_finetune_classification(self):
        df = load_iris(as_frame=True).frame
        df['finetune_group'] = np.random.randint(3, size=df.shape[0])  # random splits to finetune

        accs_single_split, accs_all_data = self.finetune_helper(df,
                                                                'target',
                                                                'finetune_group',
                                                                balanced_accuracy_score,
                                                                '__test_finetune_categorical')
        assert np.array(accs_single_split).mean() >= 0.8
        assert np.array(accs_all_data).mean() >= 0.8
