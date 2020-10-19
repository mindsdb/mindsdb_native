import unittest
from mindsdb_native.libs.helpers.locking import MDBLock
import portalocker


class ExceptionForTest(BaseException): pass


def func():
    raise ExceptionForTest('random exception')


def test_exclusive_lock():
    lock = MDBLock('exclusive', 'name')

    with unittest.assertRaises(ExceptionForTest):
        lock(func)()

    with unittest.assertRaises(ExceptionForTest):
        with lock:
            func()
