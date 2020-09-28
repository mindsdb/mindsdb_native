import logging
import shutil

import pytest

from mindsdb_native.libs.controllers.transaction import Transaction
from mindsdb_native.libs.data_types.mindsdb_logger import MindsdbLogger
from mindsdb_native.libs.controllers.predictor import Predictor
from mindsdb_native.config import CONFIG


pytest_plugins = ("plugin",)


@pytest.fixture(autouse=True)
def config(tmp_path):
    CONFIG.MINDSDB_STORAGE_PATH = str(tmp_path)
    yield CONFIG
    shutil.rmtree(CONFIG.MINDSDB_STORAGE_PATH)


@pytest.fixture()
def logger():
    return MindsdbLogger(log_level=logging.DEBUG, uuid='test')


@pytest.fixture()
def session():
    return Predictor(name='test')


@pytest.fixture()
def transaction(logger):
    lmd, hmd = {}, {}

    lmd['type'] = 'foobar'
    transaction = Transaction(session=session,
                              light_transaction_metadata=lmd,
                              heavy_transaction_metadata=hmd,
                              logger=logger)
    return transaction
