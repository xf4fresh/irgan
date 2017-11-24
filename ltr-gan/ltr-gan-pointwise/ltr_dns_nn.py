import cPickle
import random
import tensorflow as tf
import numpy as np
from eval.precision import precision_at_k, new_precision_at_k
from eval.ndcg import ndcg_at_k, new_ndcg_at_k
from eval.map import MAP, new_MAP
from eval.mrr import MRR, new_MRR
import utils as ut
from dis_model_pairwise_nn import DIS

FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.0001

DNS_K = 15

workdir = '/media/dxf/ICIRA2017/LNAI_10462-10464/MQ2008-semi'
DIS_TRAIN_FILE = workdir + '/run-train-dns.txt'
DNS_MODEL_BEST_FILE = workdir + '/dns_best_nn.model'
dataset_path = workdir + '/dataset.npz'


def create_dataset():
    query_url_feature, _, _ = ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
    query_pos_train = ut.get_query_pos(workdir + '/train.txt')
    query_pos_test = ut.get_query_pos(workdir + '/test.txt')

    # modify data
    pos_train = list()
    pos_test = list()
    neg_feature = list()
    for query in query_url_feature.keys():
        pos_url_test = list()
        pos_url_train = list()
        if query in query_pos_test:
            pos_url_test = query_pos_test[query]
        if query in query_pos_train:
            pos_url_train = query_pos_train[query]

        intersection_len = len(set(pos_url_test) & set(pos_url_train))
        if intersection_len > 0:
            print '[warn] query:%s, train&test count:%d' % (query, len(set(pos_url_test) & set(pos_url_train)))

        for url in query_url_feature[query].keys():
            if url in pos_url_train:
                pos_train.append(query_url_feature[query][url])
            elif url in pos_url_test:
                pos_test.append(query_url_feature[query][url])
            else:
                neg_feature.append(query_url_feature[query][url])

    pos_train = np.array(pos_train, dtype='float32')
    pos_test = np.array(pos_test, dtype='float32')
    neg_all = np.array(neg_feature, dtype='float32')
    np.savez(dataset_path, pos_train=pos_train, pos_test=pos_test, neg_all=neg_all)


def generate_dns(sess, model, filename):
    data = []
    print('dynamic negative sampling ...')

    pos_count = pos_train_data.shape[0]
    # neg_count = neg_data.shape[0]

    candidate_score = sess.run(model.pred_score, feed_dict={model.pred_data: neg_data})

    neg_list = []
    for i in range(pos_count):
        choice_index = np.random.choice(np.arange(len(candidate_score)), size=DNS_K)  # true or false
        choice_score = np.array(candidate_score)[choice_index]
        neg_list.append(np.argmax(choice_score))

    for i in range(pos_count):
        data.append((i, neg_list[i]))

    random.shuffle(data)
    with open(filename, 'w') as fout:
        for (pos, neg) in data:
            fout.write(','.join([str(f) for f in pos_train_data[pos]]) + '\t'
                       + ','.join([str(f) for f in neg_data[neg]]) + '\n')
            fout.flush()


def main():
    discriminator = DIS(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, D_LEARNING_RATE, loss='log', param=None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('start dynamic negative sampling with log ranking discriminator')
    p_best_val = 0.0
    ndcg_best_val = 0.0

    for epoch in range(200):
        generate_dns(sess, discriminator, DIS_TRAIN_FILE)
        train_size = ut.file_len(DIS_TRAIN_FILE)
        # exit(0)

        index = 1
        while True:
            if index > train_size:
                break
            if index + BATCH_SIZE <= train_size + 1:
                input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
            else:
                input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, train_size - index + 1)
            index += BATCH_SIZE

            input_pos = np.asarray(input_pos)
            input_neg = np.asarray(input_neg)

            _ = sess.run(discriminator.d_updates, feed_dict={discriminator.pos_data: input_pos,
                                                             discriminator.neg_data: input_neg})

        p_5 = new_precision_at_k(sess, discriminator, test_features, test_labels, k=5)
        ndcg_5 = new_ndcg_at_k(sess, discriminator, pos_test_data.shape[0], test_features, test_labels, k=5)

        if p_5 > p_best_val:
            p_best_val = p_5
            discriminator.save_model(sess, DNS_MODEL_BEST_FILE)
            print("Best: ", " p@5 ", p_5, "ndcg@5 ", ndcg_5)
        elif p_5 == p_best_val:
            if ndcg_5 > ndcg_best_val:
                ndcg_best_val = ndcg_5
                discriminator.save_model(sess, DNS_MODEL_BEST_FILE)
                print("Best: ", " p@5 ", p_5, "ndcg@5 ", ndcg_5)

    sess.close()
    param_best = cPickle.load(open(DNS_MODEL_BEST_FILE))
    assert param_best is not None
    discriminator_best = DIS(FEATURE_SIZE, HIDDEN_SIZE, WEIGHT_DECAY, D_LEARNING_RATE, loss='log', param=param_best)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    p_1_best = new_precision_at_k(sess, discriminator_best, test_features, test_labels, k=1)
    p_3_best = new_precision_at_k(sess, discriminator_best, test_features, test_labels, k=3)
    p_5_best = new_precision_at_k(sess, discriminator_best, test_features, test_labels, k=5)
    p_10_best = new_precision_at_k(sess, discriminator_best, test_features, test_labels, k=10)

    ndcg_1_best = new_ndcg_at_k(sess, discriminator_best, pos_test_data.shape[0], test_features, test_labels, k=1)
    ndcg_3_best = new_ndcg_at_k(sess, discriminator_best, pos_test_data.shape[0], test_features, test_labels, k=3)
    ndcg_5_best = new_ndcg_at_k(sess, discriminator_best, pos_test_data.shape[0], test_features, test_labels, k=5)
    ndcg_10_best = new_ndcg_at_k(sess, discriminator_best, pos_test_data.shape[0], test_features, test_labels, k=10)

    map_best = new_MAP(sess, discriminator_best, test_features, test_labels)
    mrr_best = new_MRR(sess, discriminator_best, test_features, test_labels)

    print("Best ", "p@1 ", p_1_best, "p@3 ", p_3_best, "p@5 ", p_5_best, "p@10 ", p_10_best)
    print("Best ", "ndcg@1 ", ndcg_1_best, "ndcg@3 ", ndcg_3_best, "ndcg@5 ", ndcg_5_best, "p@10 ", ndcg_10_best)
    print("Best MAP ", map_best)
    print("Best MRR ", mrr_best)


if __name__ == '__main__':
    if False:
        create_dataset()
        exit(0)

    if True:
        npz_file = np.load(dataset_path)
        pos_train_data, pos_test_data, neg_data = npz_file['pos_train'], npz_file['pos_test'], npz_file['neg_all']
        np.random.shuffle(neg_data)
        print pos_train_data.shape, pos_test_data.shape, neg_data.shape
        # exit(0)

        # from sklearn.model_selection import train_test_split
        # pos_train, pos_test = train_test_split(pos_data, random_state=2017, test_size=0.2)
        # neg_train, neg_test = train_test_split(neg_data, random_state=2017, test_size=0.2)
        test_features = np.concatenate((pos_test_data, neg_data), axis=0)
        test_labels = np.array([1] * pos_test_data.shape[0] + [0] * neg_data.shape[0])

        main()
