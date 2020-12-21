import os
import unittest

from mindsdb_native import FileDS


class TestFileDS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.issued_csv = "issued.csv"
        with open(cls.issued_csv, "w") as f:
            f.write("x, y\n")
            for _ in range(20):
                f.write('"With a hard g like gift", "With a hard g like gift"\n')
                f.write('"With a hard ""g,"" like ""gift""", "With a hard ""g,"" like ""gift"""\n')

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.issued_csv)

    # test for https://github.com/mindsdb/mindsdb_native/issues/347
    def test_file_ds_with_issued_csv(self):
        ds = FileDS(self.issued_csv)
        try:
            ds._handle_source()
        except Exception as e:
            assert False, "issue csv were not handled properly: {}".format(e)


if __name__ == '__main__':
    unittest.main()
