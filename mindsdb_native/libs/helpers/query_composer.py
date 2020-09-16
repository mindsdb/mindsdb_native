def create_history_query(query, tss, type_map, row):
    group_by_filter = []
    for group_column in tss.get('group_by',[]):
        group_by_filter.append(f'{group_column}=' + str(row[group_column]))
    group_by_filter = ' AND '.join(group_by_filter)

    order_by_list = []
    for order_column in tss.get('order_by'):
        order_by_list.append(order_column)
    order_by_list = ', '.join(order_by_list)

    query = query.lower()
    # If the initial training query had a limit we must remove it
    if ' limit ' in query:
        split_query = query.split(' limit ')
        # If there is a limit statement or more
        if len(split_query) > 1:
            # If the last limit statement is not inside a sub-qeury
            if ')' not in split_query[-1]:
                query = ' limit '.join(split_query[:-1])

    # append filter
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

        query = split_query[0] + f' ORDER BY {order_by_list} DESC' + split_query[1]
    else:
        query += f' ORDER BY {order_by_list} DESC '

    # parse and append limit
    if 'window' in tss:
        limit = tss['window']
        #query += f' LIMIT 1,{limit}' <--- if we assume the last row is the one we are predicting from
        query += f' LIMIT 1,{limit}'
    else:
        raise NotImplementedError('Historical queries not yet supported for `dynamic_window`')

    return query
