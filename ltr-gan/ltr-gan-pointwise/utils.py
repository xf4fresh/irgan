import linecache


def load_all_query_url_feature(file, feature_size):
    query_url_feature = {}
    query_url_index = {}
    query_index_url = {}

    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            query = cols[1].split(':')[1]
            url = cols[-7]
            feature = []
            for i in range(2, 2 + feature_size):
                feature.append(float(cols[i].split(':')[1]))

            # if max_query is None:
            #     pass
            # else:
            #     if query == max_query:
            #         pass
            #     else:
            #         continue
            if query in query_url_feature.keys():
                query_url_feature[query][url] = feature
                query_index_url[query].append(url)
                query_url_index[query][url] = len(query_index_url[query]) - 1
            else:
                query_url_feature[query] = {url: feature}
                query_index_url[query] = [url]
                query_url_index[query] = {url: 0}
    return query_url_feature, query_url_index, query_index_url


def get_query_pos(file):
    query_pos = {}
    line_num = -1
    with open(file) as fin:
        for line in fin:
            try:
                line_num += 1
                cols = line.split()
                rank = float(cols[0])
                query = cols[1].split(':')[1]
                url = cols[-7]
                if rank > 0.0:
                    if query in query_pos:
                        query_pos[query].append(url)
                    else:
                        query_pos[query] = [url]
            except Exception, e:
                print "except in line %d" % line_num
                continue

    # # find max query
    # max_query = ''
    # max_len = 0
    # for query in query_pos:
    #     print query, len(query_pos[query])
    #     if len(query_pos[query]) > max_len:
    #         max_len = len(query_pos[query])
    #         max_query = query
    # print "max_query:%s" % max_query

    return query_pos


def get_pos(file):
    query_pos = []
    with open(file) as fin:
        for line in fin:
            cols = line.split()
            rank = float(cols[0])
            query = cols[1].split(':')[1]
            url = cols[-7]
            if rank > 0.0:
                if query in query_pos:
                    query_pos.append(url)
                else:
                    query_pos[query] = [url]

    # # find max query
    # max_query = ''
    # max_len = 0
    # for query in query_pos:
    #     print query, len(query_pos[query])
    #     if len(query_pos[query]) > max_len:
    #         max_len = len(query_pos[query])
    #         max_query = query
    # print "max_query:%s" % max_query

    return query_pos


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


# Get batch data from training set
def get_batch_data(file, index, size):
    pos = []
    neg = []
    for i in range(index, index + size):
        line = linecache.getline(file, i)
        line = line.strip().split()
        pos.append([float(x) for x in line[0].split(',')])
        neg.append([float(x) for x in line[1].split(',')])
    return pos, neg
