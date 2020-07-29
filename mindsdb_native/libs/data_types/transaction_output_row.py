import numpy as np

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.helpers.general_helpers import value_isnan

def get_important_missing_cols(lmd, prediction_row, pred_col):
    if lmd['column_importances'] is None or len(lmd['column_importances']) < 2:
        important_cols = [col for col in lmd['columns'] if col not in lmd['predict_columns'] and not col.startswith('model_')]
    else:
        top_30_val = np.percentile(list(lmd['column_importances'].values()),70)
        important_cols = [col for col in lmd['column_importances'] if lmd['column_importances'][col] >= top_30_val]

    important_missing_cols = []
    for col in important_cols:
        if col not in prediction_row or prediction_row[col] is None or str(prediction_row[col]) == ''  or str(prediction_row[col]) == 'None' or value_isnan(prediction_row[col]):
                important_missing_cols.append(col)

    return important_missing_cols

class TransactionOutputRow:
    def __init__(self, transaction_output, row_index):
        self._transaction_output = transaction_output
        self._predict_columns = self._transaction_output._transaction.lmd['predict_columns']
        self._row_index = row_index
        self._col_stats = self._transaction_output._transaction.lmd['stats_v2']
        self._data = self._transaction_output._data
        self.explanation = self.explain()

    def __getitem__(self, item):
        return self._data[item][self._row_index]

    def __contains__(self, item):
        return item in self._data.keys()

    def explain(self):
        answers = {}
        for pred_col in self._predict_columns:
            answers[pred_col] = {}
            prediction_row = {col: self._data[col][self._row_index] for col in self._data.keys()}

            answers[pred_col]['predicted_value'] = prediction_row[pred_col]


            if f'{pred_col}_model_confidence' in prediction_row:
                answers[pred_col]['confidence'] = (prediction_row[f'{pred_col}_model_confidence'] * 3 + prediction_row[f'{pred_col}_confidence'] * 1)/4
            else:
                answers[pred_col]['confidence'] = prediction_row[f'{pred_col}_confidence']

            answers[pred_col]['confidence'] = round(answers[pred_col]['confidence'], 4)

            quality = 'very confident'
            if answers[pred_col]['confidence'] < 0.8:
                quality = 'confident'
            if answers[pred_col]['confidence'] < 0.6:
                quality = 'somewhat confident'
            if answers[pred_col]['confidence'] < 0.4:
                quality = 'not very confident'
            if answers[pred_col]['confidence'] < 0.2:
                quality = 'not confident'

            answers[pred_col]['prediction_quality'] = quality

            if self._col_stats[pred_col]['typing']['data_type'] in (DATA_TYPES.NUMERIC, DATA_TYPES.DATE):
                if f'{pred_col}_confidence_range' in prediction_row:
                    answers[pred_col]['confidence_interval'] = prediction_row[f'{pred_col}_confidence_range']

            important_missing_cols = get_important_missing_cols(self._transaction_output._transaction.lmd, prediction_row, pred_col)
            answers[pred_col]['important_missing_information'] = important_missing_cols

            if self._transaction_output._input_confidence is not None:
                answers[pred_col]['confidence_composition'] = {k:v for (k,v) in self._transaction_output._input_confidence[pred_col].items() if v > 0}

            if self._transaction_output._extra_insights is not None:
                answers[pred_col]['extra_insights'] = self._transaction_output._extra_insights[pred_col]

        return answers


    def summarize(self):
        answers = self.explain()
        simple_answers = []

        for pred_col in answers:
            confidence = round(answers[pred_col]['confidence']*100,2)
            value = answers[pred_col]['predicted_value']
            if 'confidence_interval' in answers[pred_col]:
                start = answers[pred_col]['confidence_interval'][0]
                end = answers[pred_col]['confidence_interval'][1]
                simple_col_answer = f'We are {confidence}% confident the value of "{pred_col}" is between {start} and {end}'
            else:
                simple_col_answer = f'We are {confidence}% confident the value of "{pred_col}" is {value}'
            simple_answers.append(simple_col_answer)

        return '* ' + '\n* '.join(simple_answers)

    def __str__(self):
        return str(self.summarize())

    def as_dict(self):
        return {key: self._data[key][self._row_index] for key in list(self._data.keys()) if not key.startswith('model_')}

    def as_list(self):
        #Note that here we will not output the confidence columns
        return [self._data[col][self._row_index] for col in list(self._data.keys()) if not col.startswith('model_')]

    def raw_predictions(self):
        return {key: self._data[key][self._row_index] for key in list(self._data.keys()) if key.startswith('model_')}
