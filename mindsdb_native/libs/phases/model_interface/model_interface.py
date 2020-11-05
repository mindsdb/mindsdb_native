from mindsdb_native.libs.phases.base_module import BaseModule
from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.phases.model_interface.lightwood_backend import LightwoodBackend

import datetime


class ModelInterface(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = None
        self.nr_predictions = self.transaction.lmd['tss']['nr_predictions']
        self.nn_mixer_only = False

    def run(self, mode='train'):
        self.transaction.model_backend = LightwoodBackend(self.transaction)

        if mode == 'train':
            self.transaction.model_backend.train()
            self.transaction.lmd['train_end_at'] = str(datetime.datetime.now())
        elif mode == 'predict':
            self.transaction.hmd['predictions'] = self.transaction.model_backend.predict()
