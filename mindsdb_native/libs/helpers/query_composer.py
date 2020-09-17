from mindsdb_native.libs.constants.mindsdb import *
import re

i_gb = re.compile(re.escape(' group by '), re.IGNORECASE)
i_ob = re.compile(re.escape(' order by '), re.IGNORECASE)
i_wh = re.compile(re.escape(' where '), re.IGNORECASE)
i_li = re.compile(re.escape(' limit '), re.IGNORECASE)

def create_history_query(query, tss, stats, row):
    group_by_filter = []
    group_by_arr = tss['group_by'] if tss['group_by'] is not None else []
    for group_column in group_by_arr:
        if stats[group_column]['typing']['data_type'] in [DATA_TYPES.TEXT,DATA_TYPES.CATEGORICAL]:
            group_by_filter.append(f'{group_column}=' + "'" + str(row[group_column]) + "'")
        else:
            group_by_filter.append(f'{group_column}=' + str(row[group_column]))

    group_by_filter = ' AND '.join(group_by_filter)

    order_by_list = []
    for order_column in tss.get('order_by'):
        order_by_list.append(order_column)
    order_by_list = ', '.join(order_by_list)

    query = query.rstrip(' ')
    query = query.rstrip(';')
    query = i_gb.sub(' group by ', query)
    query = i_ob.sub(' order by ', query)
    query = i_wh.sub(' where ', query)
    query = i_li.sub(' limit ', query)
    # If the initial training query had a limit we must remove it
    if ' limit ' in query:
        split_query = query.split(' limit ')
        # If there is a limit statement or more
        if len(split_query) > 1:
            # If the last limit statement is not inside a sub-qeury
            if ')' not in split_query[-1]:
                query = ' limit '.join(split_query[:-1])

    # append filter
    if len(group_by_filter) > 0:
        if ' where ' in query:
            split_query = query.split(' where ')
            query = split_query[0] + f' WHERE {group_by_filter} AND ' + split_query[1]
        elif ' group by ' in query:
            split_query = query.split(' group by ')
            query = split_query[0] + f' WHERE {group_by_filter} ' + split_query[1]
        elif ' having ' in query:
            split_query = query.split(' having ')
            query = split_query[0] + f' WHERE {group_by_filter} ' + split_query[1]
        elif ' order by ' in query:
            split_query = query.split(' order by ')
            query = split_query[0] + f' WHERE {group_by_filter} ' + split_query[1]
        else:
            query += f' WHERE {group_by_filter}'

    # append order
    if ' order by ' in query:
        split_query = query.split(' order by ')
        if len(split_query) > 2:
            raise NotImplementedError('Support for more than one order by not implemented in query parsing !')

        query = split_query[0] + f' ORDER BY {order_by_list} DESC'
    else:
        query += f' ORDER BY {order_by_list} DESC'

    # parse and append limit
    if 'window' in tss:
        limit = tss['window']
        #query += f' LIMIT 1,{limit}' <--- if we assume the last row is the one we are predicting from
        query += f' LIMIT {limit}'
    else:
        raise NotImplementedError('Historical queries not yet supported for `dynamic_window`')

    return query
