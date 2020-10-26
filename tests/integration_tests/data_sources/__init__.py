import os
import json

_var_name = 'DATABASE_CREDENTIALS_STRINGIFIED_JSON'
_var_value = os.getenv(var_name)

assert var_value is not None, var_name + ' ' + 'is not set'

DB_CREDENTIALS = json.loads(var_value)
