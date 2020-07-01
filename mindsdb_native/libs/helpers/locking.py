import os

import portalocker

from mindsdb_native.config import CONFIG


def learn_lock(name):
    learn_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_learn.lock')
    f = open(learn_lock_fn, 'a+')
    portalocker.lock(f, portalocker.LOCK_EX+portalocker.LOCK_NB)
    return f

def predict_lock(name):
    learn_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_learn.lock')
    f = open(learn_lock_fn, 'a+')
    portalocker.lock(f, portalocker.LOCK_SH+portalocker.LOCK_NB)
    return f

def delete_lock(name):
    learn_lock(name)
    delete_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_delete.lock')
    f = open(delete_lock_fn, 'a+')
    portalocker.lock(f, portalocker.LOCK_EX+portalocker.LOCK_NB)
    return f

def get_data_lock(name):
    delete_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_delete.lock')
    f = open(delete_lock_fn, 'a+')
    portalocker.lock(f, portalocker.LOCK_SH+portalocker.LOCK_NB)
    return f

def unlock(file):
    portalocker.unlock(file)
