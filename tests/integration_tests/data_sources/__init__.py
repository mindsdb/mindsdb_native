import os
import json

_var_name = 'DATABASE_CREDENTIALS_STRINGIFIED_JSON'
_var_value = os.getenv(_var_name)

assert _var_value is not None, _var_name + ' ' + 'is not set'

DB_CREDENTIALS = json.loads(_var_value)
