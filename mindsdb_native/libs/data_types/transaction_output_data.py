from mindsdb_native.libs.constants.mindsdb import *

from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.libs.data_types.transaction_output_row import TransactionOutputRow


class TrainTransactionOutputData():
    def __init__(self):
        self.data_frame = None
        self.columns = None


class PredictTransactionOutputData():
    def __init__(self, transaction, data):
        self._data = data
        self._transaction = transaction
        self._input_confidence = None
        self._extra_insights = None

    def __iter__(self):
        for i, value in enumerate(self._data[self._transaction.lmd['columns'][0]]):
            yield TransactionOutputRow(self, i)

    def __getitem__(self, item):
        return TransactionOutputRow(self, item)

    def __str__(self):
        return str(self._data)

    def __len__(self):
        return len(self._data[self._transaction.lmd['columns'][0]])
