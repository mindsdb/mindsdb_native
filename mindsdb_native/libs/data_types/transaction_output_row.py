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
        self.transaction_output = transaction_output
        self.predict_columns = self.transaction_output.transaction.lmd['predict_columns']
        self.row_index = row_index
        self.col_stats = self.transaction_output.transaction.lmd['stats_v2']
        self.data = self.transaction_output.data
        self.explanation = self.explain()

    def __getitem__(self, item):
        return self.data[item][self.row_index]

    def __contains__(self, item):
        return item in self.data.keys()

    def explain(self):
        answers = {}
        for pred_col in self.predict_columns:
            answers[pred_col] = {}

            prediction_row = {col: self.data[col][self.row_index] for col in self.data.keys()}

            answers[pred_col]['predicted_value'] = prediction_row[pred_col]

            if f'{pred_col}_model_confidence' in prediction_row:
                answers[pred_col]['confidence'] = round((prediction_row[f'{pred_col}_model_confidence'] * 3 + prediction_row[f'{pred_col}_confidence'] * 1)/4, 4)
            else:
                answers[pred_col]['confidence'] = prediction_row[f'{pred_col}_confidence']

            quality = 'very confident'
            if answers[pred_col]['confidence'] < 0.8:
                quality = 'confident'
            if answers[pred_col]['confidence'] < 0.6:
                quality = 'somewhat confident'
            if answers[pred_col]['confidence'] < 0.4:
                quality = 'not very confident'
            if answers[pred_col]['confidence'] < 0.2:
                quality = 'not confident'

            answers[pred_col]['explanation'] = {
                'prediction_quality': quality
            }

            if self.col_stats[pred_col]['typing']['data_type'] in (DATA_TYPES.NUMERIC, DATA_TYPES.DATE):
                if f'{pred_col}_confidence_range' in prediction_row:
                    answers[pred_col]['explanation']['confidence_interval'] = prediction_row[f'{pred_col}_confidence_range']

            important_missing_cols = get_important_missing_cols(self.transaction_output.transaction.lmd, prediction_row, pred_col)
            answers[pred_col]['explanation']['important_missing_information'] = important_missing_cols

            if self.transaction_output.input_confidence is not None:
                answers[pred_col]['explanation']['confidence_composition'] = {k:v for (k,v) in self.transaction_output.input_confidence[pred_col].items() if v > 0}

            if self.transaction_output.extra_insights is not None:
                answers[pred_col]['explanation']['extra_insights'] = self.transaction_output.extra_insights[pred_col]

            for k in answers[pred_col]['explanation']:
                answers[pred_col][k] = answers[pred_col]['explanation'][k]

        return answers


    def epitomize(self):
        answers = self.explain()
        simple_answers = []

        for pred_col in answers:
            confidence = answers[pred_col]['confidence']
            value = answers[pred_col]['predicted_value']
            simple_col_answer = f'We are {confidence}% confident the value of "{pred_col}" is {value}'
            simple_answers.append(simple_col_answer)

        return '* ' + '\n* '.join(simple_answers)

    def __str__(self):
        return str(self.epitomize())

    def as_dict(self):
        return {key: self.data[key][self.row_index] for key in list(self.data.keys()) if not key.startswith('model_')}

    def as_list(self):
        #Note that here we will not output the confidence columns
        return [self.data[col][self.row_index] for col in list(self.data.keys()) if not col.startswith('model_')]

    def raw_predictions(self):
        return {key: self.data[key][self.row_index] for key in list(self.data.keys()) if key.startswith('model_')}
