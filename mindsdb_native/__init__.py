import sys
if sys.version_info < (3,6):
    raise Exception('Sorry, For MindsDB Python < 3.6 is not supported')

# @TODO: FIND A WAY TO ACTUALLY SOLVE THIS ASAP !!!
# HORRIBLE HACK TO AVOID SEGFAULT
import lightwood
# HORRIBLE HACK TO AVOID SEGFAULT
# @TODO: FIND A WAY TO ACTUALLY SOLVE THIS ASAP !!!
from mindsdb_native.libs.controllers import functional
F = functional

from mindsdb_native.config import CONFIG
import mindsdb_native.libs.constants.mindsdb as CONST

from mindsdb_native.__about__ import __package_name__ as name, __version__

from mindsdb_native.libs.controllers.predictor import Predictor

from mindsdb_datasources import *

MindsDB = Predictor

# Wrap in try catch since we aren't running this in the CI
try:
    from mindsdb_native.libs.helpers.general_helpers import check_for_updates
    from mindsdb_native.config import CONFIG
    if CONFIG.CHECK_FOR_UPDATES and CONFIG.telemetry_enabled():
        check_for_updates()
except Exception as e:
    print(e)
