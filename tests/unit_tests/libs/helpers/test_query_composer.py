import pytest
from mindsdb_native.libs.helpers.query_composer import create_history_query

def test_query_composer():
    tss = {
        'order_by': 'A'
        ,'window': 6
        ,'group_by': None
    }
    stats = {
        'A': {'typing':{
            'data_type': 'Text'
        }}
        ,'B': {'typing':{
            'data_type': 'Numeric'
        }}
        ,'C': {'typing':{
            'data_type': 'Text'
        }}
    }

    new_query = create_history_query('SELECT * FROM table WHERE Z=55;', tss, stats, {'A': 100})
    assert new_query.lower() == 'SELECT * FROM table WHERE Z=55 ORDER BY A DESC LIMIT 6'.lower()

    tss['group_by'] = 'B'

    new_query = create_history_query('SELECT * FROM table WHERE Z=55;', tss, stats, {'A': 100, 'B': 500})
    assert new_query.lower() == 'SELECT * FROM table WHERE B=500 AND Z=55 ORDER BY A DESC LIMIT 6'.lower()

    new_query = create_history_query('SELECT * FROM table WHERE Z=55 GROUP BY W;', tss, stats, {'A': 100, 'B': 500})
    assert new_query.lower() == 'SELECT * FROM table WHERE B=500 AND Z=55 GROUP BY W ORDER BY A DESC LIMIT 6'.lower()

    new_query = create_history_query('SELECT * FROM table WHERE Z=55 GROUP BY W LIMIT 500;', tss, stats, {'A': 100, 'B': 500})
    assert new_query.lower() == 'SELECT * FROM table WHERE B=500 AND Z=55 GROUP BY W ORDER BY A DESC LIMIT 6'.lower()

    tss['group_by'] = ['B','C']
    new_query = create_history_query('SELECT * FROM table WHERE Z=55 GROUP BY W LIMIT 500;', tss, stats, {'A': 100, 'B': 500, 'C': 'value of C'})
    assert new_query.lower() == 'SELECT * FROM table WHERE B=500 AND C=\'value of C\' AND Z=55 GROUP BY W ORDER BY A DESC LIMIT 6'.lower()

    new_query = create_history_query('SELECT * FROM table WHERE Z=55 GROUP BY W ORDER BY N LIMIT 500;', tss, stats, {'A': 100, 'B': 500, 'C': 'value of C'})
    assert new_query.lower() == 'SELECT * FROM table WHERE B=500 AND C=\'value of C\' AND Z=55 GROUP BY W ORDER BY A DESC LIMIT 6'.lower()

    tss['group_by'] = ['C']
    tss['order_by'] = ['A','B']
    new_query = create_history_query('SELECT * FROM table WHERE Z=55 GROUP BY W ORDER BY N LIMIT 500;', tss, stats, {'A': 100, 'B': 500, 'C': 'value of C'})
    assert new_query.lower() == 'SELECT * FROM table WHERE C=\'value of C\' AND Z=55 GROUP BY W ORDER BY A, B DESC LIMIT 6'.lower()
