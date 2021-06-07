from collections import defaultdict
from copy import deepcopy

import pandas as pd

from mindsdb_native.config import CONFIG
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.data_types.mindsdb_logger import log


class DataSplitter(BaseModule):
    def run(self):
        group_by = self.transaction.lmd['tss']['group_by'] or []

        NO_GROUP = tuple()

        all_indexes = defaultdict(list)

        for i, row in self.transaction.input_data.data_frame.iterrows():
            all_indexes[NO_GROUP].append(i)

        if len(group_by) > 0:
            for i, row in self.transaction.input_data.data_frame.iterrows():
                all_indexes[tuple(row[group_by])].append(i)

        train_indexes = defaultdict(list)
        test_indexes = defaultdict(list)
        validation_indexes = defaultdict(list)

        # move indexes to corresponding train, test, validation, etc and trim input data accordingly
        if self.transaction.lmd['type'] == TRANSACTION_LEARN:
            data_split_indexes = self.transaction.lmd.get('data_split_indexes')
            if data_split_indexes is not None:
                train_indexes[NO_GROUP] = data_split_indexes['train_indexes']
                test_indexes[NO_GROUP] = data_split_indexes['test_indexes']
                validation_indexes[NO_GROUP] = data_split_indexes['validation_indexes']
            else:

                if len(group_by) > 0:
                    for group in all_indexes:
                        if group == NO_GROUP: continue
                        length = len(all_indexes[group])
                        # for time series, val and test should both have >= 2*nr_predictions rows
                        train_cutoff = max(length * 2 * CONFIG.TEST_TRAIN_RATIO,
                                           2 * 2 * self.transaction.lmd['tss'].get('nr_predictions', 0))

                        train_a = 0
                        train_b = round(length - train_cutoff)
                        train_indexes[group] = all_indexes[group][train_a:train_b]

                        test_a = train_b
                        test_b = train_b + round(train_cutoff / 2)
                        test_indexes[group] = all_indexes[group][test_a:test_b]

                        valid_a = test_b
                        valid_b = length
                        validation_indexes[group] = all_indexes[group][valid_a:valid_b]

                        train_indexes[NO_GROUP].extend(train_indexes[group])
                        test_indexes[NO_GROUP].extend(test_indexes[group])
                        validation_indexes[NO_GROUP].extend(validation_indexes[group])
                else:
                    length = len(all_indexes[NO_GROUP])
                    train_cutoff = max(length * 2 * CONFIG.TEST_TRAIN_RATIO,
                                       2 * 2 * self.transaction.lmd['tss'].get('nr_predictions', 0))

                    # make sure that the last in the time series are also the subset used for test
                    train_a = 0
                    train_b = round(length - train_cutoff)
                    train_indexes[NO_GROUP] = all_indexes[NO_GROUP][train_a:train_b]

                    valid_a = train_b
                    valid_b = train_b + round(train_cutoff / 2)
                    validation_indexes[NO_GROUP] = all_indexes[NO_GROUP][valid_a:valid_b]

                    test_a = valid_b
                    test_b = length
                    test_indexes[NO_GROUP] = all_indexes[NO_GROUP][test_a:test_b]

            self.transaction.input_data.train_df = self.transaction.input_data.data_frame.loc[train_indexes[NO_GROUP]].copy()
            self.transaction.input_data.test_df = self.transaction.input_data.data_frame.loc[test_indexes[NO_GROUP]].copy()
            self.transaction.input_data.validation_df = self.transaction.input_data.data_frame.loc[validation_indexes[NO_GROUP]].copy()

            if self.transaction.lmd['tss']['is_timeseries']:
                ts_train_row_count = len(self.transaction.input_data.train_df)
                ts_test_row_count = len(self.transaction.input_data.test_df)
                ts_val_row_count = len(self.transaction.input_data.validation_df)

                historical_train = deepcopy(self.transaction.input_data.train_df)
                historical_train['make_predictions'] = [False] * len(historical_train)

                historical_test = deepcopy(self.transaction.input_data.test_df)
                historical_test['make_predictions'] = [False] * len(historical_test)

                self.transaction.input_data.train_df['make_predictions'] = [True] * len(self.transaction.input_data.train_df)

                self.transaction.input_data.test_df['make_predictions'] = [True] * len(self.transaction.input_data.test_df)
                self.transaction.input_data.test_df = pd.concat([self.transaction.input_data.test_df,historical_train])

                self.transaction.input_data.validation_df['make_predictions'] = [True] * len(self.transaction.input_data.validation_df)
                self.transaction.input_data.validation_df = pd.concat([self.transaction.input_data.validation_df,historical_test,deepcopy(historical_train)])

            self.transaction.input_data.data_frame = None

            self.transaction.lmd['data_preparation']['test_row_count'] = len(self.transaction.input_data.test_df)
            self.transaction.lmd['data_preparation']['train_row_count'] = len(self.transaction.input_data.train_df)
            self.transaction.lmd['data_preparation']['validation_row_count'] = len(self.transaction.input_data.validation_df)

            if self.transaction.lmd['tss']['is_timeseries']:
                # restore original row count (prior to added historical context)
                self.transaction.lmd['data_preparation']['train_row_count'] = ts_train_row_count
                self.transaction.lmd['data_preparation']['validation_row_count'] = ts_val_row_count
                self.transaction.lmd['data_preparation']['test_row_count'] = ts_test_row_count

            data = {
                'subsets': [
                    [len(self.transaction.input_data.train_df), 'Train'],
                    [len(self.transaction.input_data.test_df), 'Test'],
                    [len(self.transaction.input_data.validation_df), 'Validation']
                ],
                'label': 'Number of rows per subset'
            }

            self.log.info('We have split the input data into:')
            self.log.infoChart(data, type='pie')

        return all_indexes, train_indexes, test_indexes, validation_indexes
