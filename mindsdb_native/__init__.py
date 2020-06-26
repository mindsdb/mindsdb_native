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
from mindsdb_native.libs.data_sources.maria_ds import MariaDS
from mindsdb_native.libs.data_sources.mysql_ds import MySqlDS

# Data Sources
from mindsdb_native.libs.data_sources.file_ds import FileDS
# These might not initialized properly since they require optional dependencies, so we wrap them in a try-except and don't export them if the dependencies aren't installed
try:
    from mindsdb_native.libs.data_sources.s3_ds import S3DS
except:
    pass

try:
    from mindsdb_native.libs.data_sources.postgres_ds import PostgresDS
except:
    print("PostgresDS Datasource is not available by default. If you wish to use it, please install psycopg2 or mindsdb[extra_data_sources]")



from mindsdb_native.libs.data_sources.clickhouse_ds import ClickhouseDS



MindsDB = Predictor
