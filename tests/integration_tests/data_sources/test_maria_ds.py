import pytest
import mysql.connector
import datetime
import logging
from mindsdb_native import Predictor
from mindsdb_native.libs.constants.mindsdb import DATA_TYPES, DATA_SUBTYPES
from mindsdb_native import MariaDS
from mindsdb_native import F


@pytest.mark.integration
def test_maria_ds():
    HOST = 'localhost'
    USER = 'root'
    PASSWORD = ''
    DATABASE = 'mysql'
    PORT = 4306

    con = mysql.connector.connect(host=HOST,
                                  port=PORT,
                                  user=USER,
                                  password=PASSWORD,
                                  database=DATABASE)
    cur = con.cursor()

    cur.execute('DROP TABLE IF EXISTS test_mindsdb')
    cur.execute("""CREATE TABLE test_mindsdb (
                                col_int BIGINT,
                                col_float FLOAT,
                                col_categorical Text,
                                col_bool BOOL,
                                col_text Text,
                                col_date DATE,
                                col_datetime DATETIME,
                                col_timestamp TIMESTAMP,
                                col_time TIME
                                )
                                """)
    for i in range(0, 200):
        dt = datetime.datetime.now() - datetime.timedelta(days=i)

        query = f"""INSERT INTO test_mindsdb (col_int,
                                col_float,
                                col_categorical,
                                col_bool,
                                col_text,
                                col_date,
                                col_datetime,
                                col_timestamp,
                                col_time)
                                VALUES (%s, %s,  %s,  %s,  %s, %s, %s, %s, %s)
                                """
        ci = i % 5
        values = (
            i,
            i + 0.01,
            f"Cat {ci}",
            i % 2 == 0,
            f"long long long text {i}",
            dt.date(),
            dt,
            dt.strftime('%Y-%m-%d %H:%M:%S.%f'),
            dt.strftime('%H:%M:%S.%f')
        )
        cur.execute(query, values)
    con.commit()
    con.close()

    maria_ds = MariaDS(table='test_mindsdb', host=HOST, user=USER,
                       password=PASSWORD, database=DATABASE, port=PORT)
    assert (len(maria_ds._df) == 200)

    mdb = Predictor(name='analyse_dataset_test_predictor', log_level=logging.ERROR)
    model_data = F.analyse_dataset(from_data=maria_ds)
    analysis = model_data['data_analysis_v2']
    assert model_data
    assert analysis

    def assert_expected_type(column_typing, expected_type, expected_subtype):
        assert column_typing['data_type'] == expected_type
        assert column_typing['data_subtype'] == expected_subtype
        assert column_typing['data_type_dist'][expected_type] == 200
        assert column_typing['data_subtype_dist'][expected_subtype] == 200


    assert_expected_type(analysis['col_categorical']['typing'], DATA_TYPES.CATEGORICAL, DATA_SUBTYPES.MULTIPLE)
    assert_expected_type(analysis['col_bool']['typing'], DATA_TYPES.CATEGORICAL, DATA_SUBTYPES.SINGLE)
    assert_expected_type(analysis['col_int']['typing'], DATA_TYPES.NUMERIC, DATA_SUBTYPES.INT)
    assert_expected_type(analysis['col_float']['typing'], DATA_TYPES.NUMERIC, DATA_SUBTYPES.FLOAT)
    assert_expected_type(analysis['col_date']['typing'], DATA_TYPES.DATE, DATA_SUBTYPES.DATE)
    assert_expected_type(analysis['col_datetime']['typing'], DATA_TYPES.DATE, DATA_SUBTYPES.TIMESTAMP)
    assert_expected_type(analysis['col_timestamp']['typing'], DATA_TYPES.DATE, DATA_SUBTYPES.TIMESTAMP)

    # Subtype is expected to be either .SHORT or .RICH
    try:
        assert_expected_type(analysis['col_text']['typing'], DATA_TYPES.TEXT, DATA_SUBTYPES.SHORT)
    except AssertionError:
        assert_expected_type(analysis['col_text']['typing'], DATA_TYPES.TEXT, DATA_SUBTYPES.RICH)


    # @TODO Timedeltas not supported yet
    # assert_expected_type((analysis['col_time']['typing'], DATA_TYPES.DATE, DATA_SUBTYPES.TIMEDELTA)
