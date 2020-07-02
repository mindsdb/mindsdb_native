import pytest
from mindsdb_native.libs.helpers.locking import MDBLock
import portalocker


class ExceptionForTest(BaseException): pass


def func():
    raise ExceptionForTest('random exception')


def test_locking():
    for flags in ['shared', 'exclusive']:
        lock = MDBLock(flags, 'name')

        with pytest.raises(ExceptionForTest):
            lock(func)()

        with pytest.raises(ExceptionForTest):
            with lock:
                func()
