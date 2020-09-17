import pytest
from mindsdb_native.libs.helpers.query_composer import create_history_query

def test_query_composer():
    tss = {
        'order_by': 'A'
        ,'window': 6
    }
    query = 'SELECT * FROM table WHERE Z=55;'
    stats = {
        'A': {'typing':{
            'data_type': 'TEXT'
        }}
        ,'B': {'typing':{
            'data_type': 'INT'
        }}
        ,'C': {'typing':{
            'data_type': 'FLOAT'
        }}
    }
    new_query = create_history_query(query, tss, stats, {'A': 100})

    assert new_query == 'SELECT * FROM table WHERE Z=55 ORDER BY A DESC LIMIT 6'


test_query_composer()
