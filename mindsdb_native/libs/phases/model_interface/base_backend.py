

class BaseBackend:
    def __init__(self, transaction):
        self.transaction = transaction

    def train(self):
        raise NotImplementedError

    def predict(self, mode='predict', ignore_columns=[]):
        raise NotImplementedError
