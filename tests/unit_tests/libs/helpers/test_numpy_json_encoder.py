import json
import numpy as np
from mindsdb_native.libs.helpers.general_helpers import NumpyJSONEncoder


def test_numpy_json_encoder():
    x = {'x': np.float32(5)}

    json_str = json.dumps(x, cls=NumpyJSONEncoder)
    json_str = json_str.replace('\n', '')
    json_str = json_str.replace(' ', '')
    json_str = json_str.replace('\t', '')

    assert json_str == '{"x":5.0}'
