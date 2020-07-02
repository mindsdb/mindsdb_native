import os
import sys
import inspect
import traceback
import portalocker

from mindsdb_native.config import CONFIG


def mdb_lock(flags, lock_name, argname=None):
    """
    :param lock_name: str, name of the lock
    :argname: str, name of the positional/keyword argument of
    the input function that will be concatenated with :lock_name:
    """
    if flags == 'shared':
        flags = portalocker.LOCK_SH + portalocker.LOCK_NB
    elif flags == 'exclusive':
        flags = portalocker.LOCK_EX + portalocker.LOCK_NB
    else:
        raise ValueError('expected flags to be "shared" or "exclusive"')

    def wrapper1(func):
        def wrapper2(*args, **kwargs):
            if argname is None:
                final_lock_name = '{}.lock'.format(lock_name)
            else:
                if argname.startswith('self.'):
                    assert len(argname.split('.') == 2)
                    argval = getattr(args[0], argname.lstrip('self.'))
                elif argname in kwargs:
                    argval = kwargs[argname]
                else:
                    index = inspect.getfullargspec(func).args.index(argname)
                    if index < len(args):
                        argval = args[index]
                    else:
                        raise ValueError('argname wasn\'t found in *args/**kwargs')
                final_lock_name = '{}_{}.lock'.format(lock_name, argval)

            path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, final_lock_name)
            f = open(path, 'a+')
            portalocker.lock(f, flags)

            try:
                ret = func(*args, **kwargs)
            except BaseException:
                print(traceback.format_exc())
                raise
            else:
                return ret
            finally:
                portalocker.unlock(f)

        return wrapper2
    return wrapper1
