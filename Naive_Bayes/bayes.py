# coding: utf-8
from numpy import *


def load_dataset():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]

    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    return_vec = zeros((1, len(vocab_list)))[0]
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return return_vec


def train_NBO(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = log(p1_num / p1_denom)
    p0_vect = log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


def classify_NB(vec2classify, p0_vec, p1_vec, p_class):
    p1 = sum(vec2classify * p1_vec) + log(p_class)
    p0 = sum(vec2classify * p0_vec) + log(1 - p_class)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_NB():
    list_posts, list_classes = load_dataset()
    my_vocab_list = create_vocab_list(list_posts)
    train_mat = [set_of_words2vec(my_vocab_list, post_in_doc) for post_in_doc in list_posts]
    p0_v, p1_v, p_ab = train_NBO(train_mat, list_classes)

    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print test_entry, 'classified as: ', classify_NB(this_doc, p0_v, p1_v, p_ab)

    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print test_entry, 'classified as: ', classify_NB(this_doc, p0_v, p1_v, p_ab)


testing_NB()
