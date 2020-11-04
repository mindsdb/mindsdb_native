import json
import unittest
import numpy as np
from mindsdb_native.libs.helpers.general_helpers import NumpyJSONEncoder


class TestNumpyJsonEncoder(unittest.TestCase):
    def test_numpy_json_encoder(self):
        x = {'x': np.float32(5)}

        json_str = json.dumps(x, cls=NumpyJSONEncoder)
        json_str = json_str.replace('\n', '')
        json_str = json_str.replace(' ', '')
        json_str = json_str.replace('\t', '')

        assert json_str == '{"x":5.0}'
