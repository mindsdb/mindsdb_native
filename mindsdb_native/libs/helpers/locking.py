import os
import sys
import inspect

import portalocker

from mindsdb_native.config import CONFIG


def mdb_lock(flags, lock_name, argname=None):
    """
    :param lock_name: str, name of the lock
    :argname: str, name of the positional/keyword argument of
    the input function that will be concatenated with :lock_name:
    """
    def wrapper1(func):
        def wrapper2(*args, **kwargs):
            print(func, args, kwargs)
            print(inspect.signature(func).parameters)

            if argname is None:
                final_lock_name = '{}.lock'.format(lock_name)
            else:
                if argname.startswith('self.'):
                    argval = getattr(args[0, argname.split('.')[1]])
                if argname in kwargs:
                    argval = kwargs[argname]
                else:
                    params = inspect.signature(func).parameters
                    if argname in params:
                        index = list(params).index(argname)
                        argval = args[index]
                    else:
                        raise Exception('argname wan\'t found in *args/**kwargs')
                final_lock_name = '{}_{}.lock'.format(lock_name, argval)

            path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, final_lock_name)
            f = open(path, 'a+')
            portalocker.lock(f, flags)

            try:
                ret = func(*args, **kwargs)
            except BaseException:
                portalocker.unlock(f)
                print(traceback.format_exc())
                sys.exit(1)
            else:
                portalocker.unlock(file)
                return ret

        return wrapper2
    return wrapper1


# def learn_lock(name):
#     learn_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_learn.lock')
#     f = open(learn_lock_fn, 'a+')
#     portalocker.lock(f, portalocker.LOCK_EX+portalocker.LOCK_NB)
#     return f

# def predict_lock(name):
#     learn_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_learn.lock')
#     f = open(learn_lock_fn, 'a+')
#     portalocker.lock(f, portalocker.LOCK_SH+portalocker.LOCK_NB)
#     return f

# def delete_lock(name):
#     learn_lock(name)
#     delete_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_delete.lock')
#     f = open(delete_lock_fn, 'a+')
#     portalocker.lock(f, portalocker.LOCK_EX+portalocker.LOCK_NB)
#     return f

# def get_data_lock(name):
#     delete_lock_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,f'{name}_delete.lock')
#     f = open(delete_lock_fn, 'a+')
#     portalocker.lock(f, portalocker.LOCK_SH+portalocker.LOCK_NB)
#     return f

# def unlock(file):
#     portalocker.unlock(file)
