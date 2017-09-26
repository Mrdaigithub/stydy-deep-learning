# coding: utf-8
from numpy import *
from math import log
import pickle


def get_training_lenses_data(filename):
    fr = open(filename)
    source_data = fr.readlines()
    _lenses_data = [i.strip().split('\t') for i in source_data]
    _lenses_labels = ['age', 'spectacle_prescription', 'astigmatic', 'tear_production_rate']
    return _lenses_data, _lenses_labels


def cale_entropy(dataset):
    num_dataset = len(dataset)
    label_count = {}
    for i in range(num_dataset):
        if dataset[i][-1] not in label_count:
            label_count[dataset[i][-1]] = 0
        label_count[dataset[i][-1]] += 1
    entropy = 0.0
    for i in label_count:
        prob = float(label_count[i]) / float(num_dataset)
        entropy -= prob * log(prob, 2)
    return entropy


def split_dataset(dataset, axis, value):
    ret_dataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            arr = featVec[:axis]
            arr.extend(featVec[axis + 1:])
            ret_dataset.append(arr)
    return ret_dataset


def choose_best_feature_to_split(dataset):
    num_feature = len(dataset[0]) - 1
    base_entropy = cale_entropy(dataset)
    info_gain_list = {}
    for i in range(num_feature):
        feature_list = [example[i] for example in dataset]
        unique_feature_list = set(feature_list)
        new_entropy = 0.0
        for j in unique_feature_list:
            sub_dataset = split_dataset(dataset, i, j)
            new_entropy += cale_entropy(sub_dataset)
        info_gain_list[i] = base_entropy - new_entropy
    return sorted(info_gain_list.iteritems(), key=lambda d: d[1], reverse=True)[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(labels) == 0:
        return dataset[0][-1]
    _labels = labels[:]
    current_feature = choose_best_feature_to_split(dataset)
    values = set([i[current_feature] for i in dataset])
    current_feature_val = _labels.pop(current_feature)
    tree = {current_feature_val: {}}
    for value in values:
        sub_dataset = split_dataset(dataset, current_feature, value)
        tree[current_feature_val][value] = create_tree(sub_dataset, _labels)
    return tree


def store_tree(input_tree, filename='classifier_storage.txt'):
    with open(filename, 'w') as f:
        pickle.dump(input_tree, f)


def fetch_tree(filename='classifier_storage.txt'):
    with open(filename, 'r') as f:
        return pickle.load(f)


def classify(input_tree, feat_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label
