import unittest
import mindsdb_native
from mindsdb_native import Predictor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


class TestFinetune(unittest.TestCase):
    def test_finetune(self):
        df = pd.read_csv("https://s3.eu-west-2.amazonaws.com/mindsdb-example-data/home_rentals.csv")
        target = 'rental_price'
        grouped_col = 'neighborhood'
        test_size=0.2
        groups = df[grouped_col].unique()
        data_tuples = []
        for group in groups:
            grouped = df[df[grouped_col] == group]
            train = grouped[:int(len(grouped)*(1-test_size))]
            test = grouped[int(len(grouped)*(1-test_size)):]
            if len(data_tuples) > 0:
                all_test = test.append(data_tuples[-1][1])  # add previous groups to the latest test DF
            else:
                all_test = test
            data_tuples.append((train, test, all_test))

        predictor = Predictor(name='home_rentals_finetune')

        results_single_test = []
        results_all_test = []
        for i, (train, test, all_test) in enumerate(data_tuples):
            if i == 0:
                predictor.learn(from_data=train,
                                to_predict=target,
                                stop_training_in_x_seconds=10,
                                advanced_args={'use_mixers': ['NnMixer']},
                                use_gpu=False)
            else:
                # finetune here
                predictor.adjust(from_data=train)  #  TODO: maybe add interface here for additional options

            result_single = predictor.test(when_data=test, accuracy_score_functions=r2_score)
            print(result_single)
            assert result_single['rental_price_accuracy'] >= 0.9
            result_all = predictor.test(when_data=all_test, accuracy_score_functions=r2_score)
            print(result_all)
            assert result_all['rental_price_accuracy'] >= 0.9

            results_single_test.append(result_single[f'{target}_accuracy'])
            results_all_test.append(result_all[f'{target}_accuracy'])

        print(np.array(results_single_test))
        print(np.array(results_all_test))

        print(np.array(results_single_test).mean())
        print(np.array(results_all_test).mean())
