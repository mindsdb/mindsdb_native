import os
import json
import time
import random
import string

_var_name = 'DATABASE_CREDENTIALS_STRINGIFIED_JSON'
_var_value = os.getenv(_var_name)
if _var_value is None:
    with open(os.path.join(os.path.expanduser("~"), '.mindsdb_credentials.json'), 'r') as fp:
        _var_value = fp.read()

assert _var_value is not None, _var_name + ' ' + 'is not set'

DB_CREDENTIALS = json.loads(_var_value)
