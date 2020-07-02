import os
import sys
import inspect
import traceback
import portalocker
from mindsdb_native.config import CONFIG


class MDBLock():
    """Can be used in a with statement or as a decorator"""

    def __init__(self, flags, name):
        """
        :param flags: "shared" or "exclusive"
        :param name: str
        """
        if flags == 'shared':
            flags = portalocker.LOCK_SH + portalocker.LOCK_NB
        elif flags == 'exclusive':
            flags = portalocker.LOCK_EX + portalocker.LOCK_NB
        else:
            raise ValueError('expected flags to be "shared" or "exclusive"')

        self._flags = flags
        self._name = name
        self._f = None

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def __enter__(self):
        path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self._name)
        self._f = open(path, 'a+')
        portalocker.lock(self._f, self._flags)
        return self._f

    def __exit__(self, type, value, traceback):
        portalocker.unlock(self._f)
