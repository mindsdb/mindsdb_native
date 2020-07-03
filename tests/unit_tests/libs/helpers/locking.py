import pytest
from mindsdb_native.libs.helpers.locking import MDBLock
import portalocker


class ExceptionForTest(BaseException): pass


def func():
    raise ExceptionForTest('random exception')


def test_exclusive_lock():
    lock = MDBLock('exclusive', 'name')

    with pytest.raises(ExceptionForTest):
        lock(func)()

    with pytest.raises(ExceptionForTest):
        with lock:
            func()
