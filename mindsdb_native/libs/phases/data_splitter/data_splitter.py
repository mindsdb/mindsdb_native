from mindsdb_native.config import CONFIG
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.data_types.mindsdb_logger import log


class DataSplitter(BaseModule):
    def run(self):
        group_by = self.transaction.lmd['model_group_by']
        if group_by is None or len(group_by) == 0:
            group_by = []
            # @TODO: Group by seems not to work on certain datasets and the values get split complete unevenly between train/test/validation
            if len(group_by) > 0:
                try:
                    self.transaction.input_data.data_frame = self.transaction.input_data.data_frame.sort_values(group_by)
                except Exception as e:
                    # If categories can't be sroted because of various issues, that's fine, no need for the prediction logic to fail
                    if len(self.transaction.lmd['model_group_by']) == 0:
                        group_by = []
                    else:
                        raise Exception(e)


        KEY_NO_GROUP_BY = '{PLEASE_DONT_TELL_ME_ANYONE_WOULD_CALL_A_COLUMN_THIS}##ALL_ROWS_NO_GROUP_BY##{PLEASE_DONT_TELL_ME_ANYONE_WOULD_CALL_A_COLUMN_THIS}'

        # create all indexes by group by, that is all the rows that belong to each group by
        all_indexes = {}
        train_indexes = {}
        test_indexes = {}
        validation_indexes = {}

        all_indexes[KEY_NO_GROUP_BY] = []
        train_indexes[KEY_NO_GROUP_BY] = []
        test_indexes[KEY_NO_GROUP_BY] = []
        validation_indexes[KEY_NO_GROUP_BY] = []
        for i, row in self.transaction.input_data.data_frame.iterrows():

            if len(group_by) > 0:
                group_by_value = '_'.join([str(row[group_by_index]) for group_by_index in [self.transaction.input_data.columns.index(group_by_col) for group_by_col in group_by]])

                if group_by_value not in all_indexes:
                    all_indexes[group_by_value] = []

                all_indexes[group_by_value] += [i]

            all_indexes[KEY_NO_GROUP_BY] += [i]

        # move indexes to corresponding train, test, validation, etc and trim input data accordingly
        if self.transaction.lmd['type'] == TRANSACTION_LEARN:
            for key in all_indexes:
                should_split_by_group = isinstance(group_by, list) and len(group_by) > 0

                #If this is a group by, skip the `KEY_NO_GROUP_BY` key
                if should_split_by_group and key == KEY_NO_GROUP_BY:
                    continue

                length = len(all_indexes[key])
                # this evals True if it should send the entire group data into test, train or validation as opposed to breaking the group into the subsets
                if should_split_by_group:
                    train_indexes[key] = all_indexes[key][0:round(length - length*CONFIG.TEST_TRAIN_RATIO)]
                    train_indexes[KEY_NO_GROUP_BY].extend(train_indexes[key])

                    test_indexes[key] = all_indexes[key][round(length - length*CONFIG.TEST_TRAIN_RATIO):int(round(length - length*CONFIG.TEST_TRAIN_RATIO) + round(length*CONFIG.TEST_TRAIN_RATIO/2))]
                    test_indexes[KEY_NO_GROUP_BY].extend(test_indexes[key])

                    validation_indexes[key] = all_indexes[key][(round(length - length*CONFIG.TEST_TRAIN_RATIO) + round(length*CONFIG.TEST_TRAIN_RATIO/2)):]
                    validation_indexes[KEY_NO_GROUP_BY].extend(validation_indexes[key])

                else:
                    # make sure that the last in the time series are also the subset used for test

                    train_window = (0,int(length*(1-2*CONFIG.TEST_TRAIN_RATIO)))
                    train_indexes[key] = all_indexes[key][train_window[0]:train_window[1]]
                    validation_window = (train_window[1],train_window[1] + int(length*CONFIG.TEST_TRAIN_RATIO))
                    test_window = (validation_window[1],length)
                    test_indexes[key] = all_indexes[key][test_window[0]:test_window[1]]
                    validation_indexes[key] = all_indexes[key][validation_window[0]:validation_window[1]]


            self.transaction.input_data.train_df = self.transaction.input_data.data_frame.loc[train_indexes[KEY_NO_GROUP_BY]].copy()
            self.transaction.input_data.test_df = self.transaction.input_data.data_frame.loc[test_indexes[KEY_NO_GROUP_BY]].copy()
            self.transaction.input_data.validation_df = self.transaction.input_data.data_frame.loc[validation_indexes[KEY_NO_GROUP_BY]].copy()

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

        # Initialize this here, will be overwritten if `equal_accuracy_for_all_output_categories` is specified to be True in order to account for it
        self.transaction.lmd['weight_map'] = self.transaction.lmd['output_categories_importance_dictionary']

        # Un-bias dataset for training
        for column in self.transaction.lmd['predict_columns']:
            if (self.transaction.lmd['stats_v2'][column]['typing']['data_type'] == DATA_TYPES.CATEGORICAL
                    and self.transaction.lmd['equal_accuracy_for_all_output_categories'] is True
                    and self.transaction.lmd['type'] == TRANSACTION_LEARN):

                occurance_map = {}
                ciclying_map = {}

                for i in range(0, len(self.transaction.lmd['stats_v2'][column]['histogram']['x'])):
                    ciclying_map[self.transaction.lmd['stats_v2'][column]['histogram']['x'][i]] = 0
                    occurance_map[self.transaction.lmd['stats_v2'][column]['histogram']['x'][i]] = \
                    self.transaction.lmd['stats_v2'][column]['histogram']['y'][i]

                max_val_occurances = max(occurance_map.values())

                if self.transaction.hmd['model_backend'] in ('lightwood'):
                    lightwood_weight_map = {}
                    for val in occurance_map:
                        lightwood_weight_map[val] = 1 / occurance_map[val]  # sum(occurance_map.values())

                        if column in self.transaction.lmd['output_categories_importance_dictionary']:
                            if val in self.transaction.lmd['output_categories_importance_dictionary'][column]:
                                lightwood_weight_map[val] = \
                                self.transaction.lmd['output_categories_importance_dictionary'][column][val]
                            elif '<default>' in self.transaction.lmd['output_categories_importance_dictionary'][
                                column]:
                                lightwood_weight_map[val] = \
                                self.transaction.lmd['output_categories_importance_dictionary'][column]['<default>']

                    self.transaction.lmd['weight_map'][column] = lightwood_weight_map

                # print(self.transaction.lmd['weight_map'])
                column_is_weighted_in_train = column in self.transaction.lmd['weight_map']

                if column_is_weighted_in_train:
                    dfs = ['input_data.validation_df']
                else:
                    dfs = ['input_data.train_df', 'input_data.test_df', 'input_data.validation_df']

                total_len = (len(self.transaction.input_data.train_df) + len(self.transaction.input_data.test_df) + len(self.transaction.input_data.validation_df))
                # Since pandas doesn't support append in-place we'll just do some eval-based hacks

                for dfn in dfs:
                    max_val_occurances_in_set = int(round(max_val_occurances * len(eval(dfn)) / total_len))
                    for val in occurance_map:
                        valid_rows = eval(dfn)[eval(dfn)[column] == val]
                        if len(valid_rows) == 0:
                            continue

                        appended_times = 0
                        while max_val_occurances_in_set > len(valid_rows) * (2 + appended_times):
                            exec(f'{dfn} = {dfn}.append(valid_rows)')
                            appended_times += 1

                        if int(max_val_occurances_in_set - len(valid_rows) * (1 + appended_times)) > 0:
                            exec(
                                f'{dfn} = {dfn}.append(valid_rows[0:int(max_val_occurances_in_set - len(valid_rows) * (1 + appended_times))])')

