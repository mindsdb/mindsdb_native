import unittest
from mindsdb_native.libs.helpers.locking import MDBLock


class ExceptionForTest(BaseException): pass


def func():
    raise ExceptionForTest('random exception')


class TestLocking(unittest.TestCase):
    def test_exclusive_lock(self):
        lock = MDBLock('exclusive', 'name')

        try:
            lock(func)()
        except ExceptionForTest:
            pass
        else:
            raise AssertionError

        try:
            with lock:
                func()
        except ExceptionForTest:
            pass
        else:
            raise AssertionError
