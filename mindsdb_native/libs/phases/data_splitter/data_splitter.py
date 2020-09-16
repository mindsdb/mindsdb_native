from collections import defaultdict

from mindsdb_native.config import CONFIG
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.data_types.mindsdb_logger import log


class DataSplitter(BaseModule):
    def run(self):
        group_by = self.transaction.lmd['tss']['group_by']

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
                for group in all_indexes:

                    length = len(all_indexes[group])

                    if len(group_by) > 0:
                        if group == NO_GROUP: continue

                        train_a = 0
                        train_b = round(length - length * CONFIG.TEST_TRAIN_RATIO)
                        train_indexes[group] = all_indexes[group][train_a:train_b]
                        train_indexes[NO_GROUP].extend(train_indexes[group])

                        test_a = round(length - length * CONFIG.TEST_TRAIN_RATIO)
                        test_b = round(length - length * CONFIG.TEST_TRAIN_RATIO) + round(length * CONFIG.TEST_TRAIN_RATIO / 2)
                        test_indexes[group] = all_indexes[group][test_a:test_b]
                        test_indexes[NO_GROUP].extend(test_indexes[group])

                        valid_a = (round(length - length * CONFIG.TEST_TRAIN_RATIO) + round(length * CONFIG.TEST_TRAIN_RATIO / 2))
                        valid_b = length
                        validation_indexes[group] = all_indexes[group][valid_a:valid_b]
                        validation_indexes[NO_GROUP].extend(validation_indexes[group])
                    else:
                        # make sure that the last in the time series are also the subset used for test
                        train_window = (0, int(length * (1 - 2 * CONFIG.TEST_TRAIN_RATIO)))
                        train_indexes[key] = all_indexes[key][train_window[0]:train_window[1]]
                        validation_window = (train_window[1], train_window[1] + int(length * CONFIG.TEST_TRAIN_RATIO))
                        test_window = (validation_window[1], length)
                        test_indexes[key] = all_indexes[key][test_window[0]:test_window[1]]
                        validation_indexes[key] = all_indexes[key][validation_window[0]:validation_window[1]]

            self.transaction.input_data.train_df = self.transaction.input_data.data_frame.loc[train_indexes[NO_GROUP]].copy()
            self.transaction.input_data.test_df = self.transaction.input_data.data_frame.loc[test_indexes[NO_GROUP]].copy()
            self.transaction.input_data.validation_df = self.transaction.input_data.data_frame.loc[validation_indexes[NO_GROUP]].copy()

            self.transaction.input_data.data_frame = None

            self.transaction.lmd['data_preparation']['test_row_count'] = len(self.transaction.input_data.test_df)
            self.transaction.lmd['data_preparation']['train_row_count'] = len(self.transaction.input_data.train_df)
            self.transaction.lmd['data_preparation']['validation_row_count'] = len(self.transaction.input_data.validation_df)

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
