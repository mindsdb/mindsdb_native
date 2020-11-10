import unittest
from mindsdb_native.libs.helpers.query_composer import create_history_query


class TestQueryComposer(unittest.TestCase):
    def test_query_composer(self):
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

        new_query = create_history_query('select * from table where z=55;', tss, stats, {'A': 100})
        assert new_query.lower() == 'select * from table where z=55 order by A DESC limit 6'.lower()

        tss['group_by'] = 'B'

        new_query = create_history_query('select * from table where z=55;', tss, stats, {'A': 100, 'B': 500})
        assert new_query.lower() == 'select * from table where B=500 AND z=55 order by A DESC limit 6'.lower()

        new_query = create_history_query('select * from table where z=55 group by W;', tss, stats, {'A': 100, 'B': 500})
        assert new_query.lower() == 'select * from table where B=500 AND z=55 group by W order by A DESC limit 6'.lower()

        new_query = create_history_query('select * from table where z=55 group by W limit 500;', tss, stats, {'A': 100, 'B': 500})
        assert new_query.lower() == 'select * from table where B=500 AND z=55 group by W order by A DESC limit 6'.lower()

        tss['group_by'] = ['B','C']
        new_query = create_history_query('select * from table where z=55 group by W limit 500;', tss, stats, {'A': 100, 'B': 500, 'C': 'value of C'})
        assert new_query.lower() == 'select * from table where B=500 AND C=\'value of C\' AND z=55 group by W order by A DESC limit 6'.lower()

        new_query = create_history_query('select * from table where z=55 group by W order by N limit 500;', tss, stats, {'A': 100, 'B': 500, 'C': 'value of C'})
        assert new_query.lower() == 'select * from table where B=500 AND C=\'value of C\' AND z=55 group by W order by A DESC limit 6'.lower()

        tss['group_by'] = ['C']
        tss['order_by'] = ['A','B']
        new_query = create_history_query('select * from table where z=55 group by W order by N limit 500;', tss, stats, {'A': 100, 'B': 500, 'C': 'value of C'})
        assert new_query.lower() == 'select * from table where B<500 AND C=\'value of C\' AND z=55 group by W order by A, B DESC limit 6'.lower()
