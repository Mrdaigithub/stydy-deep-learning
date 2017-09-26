# coding: utf-8
from numpy import *
from math import log
import pickle


def create_dataset():
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


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

    # numFeatures = len(dataset[0]) - 1  # the last column is used for the labels
    # baseEntropy = cale_entropy(dataset)
    # bestInfoGain = 0.0;
    # bestFeature = -1
    # for i in range(numFeatures):  # iterate over all the features
    #     featList = [example[i] for example in dataset]  # create a list of all the examples of this feature
    #     uniqueVals = set(featList)  # get a set of unique values
    #     newEntropy = 0.0
    #     for value in uniqueVals:
    #         subDataSet = split_dataset(dataset, i, value)
    #         prob = len(subDataSet) / float(len(dataset))
    #         newEntropy += prob * cale_entropy(subDataSet)
    #     infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
    #     if (infoGain > bestInfoGain):  # compare this to the best gain so far
    #         bestInfoGain = infoGain  # if better than current best, set to best
    #         bestFeature = i
    # return bestFeature  # returns an integer


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
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

    # classList = [example[-1] for example in dataset]
    # if classList.count(classList[0]) == len(classList):
    #     return classList[0]  # stop splitting when all of the classes are equal
    # if len(dataset[0]) == 1:  # stop splitting when there are no more features in dataSet
    #     return majority_cnt(classList)
    # bestFeat = choose_best_feature_to_split(dataset)
    # bestFeatLabel = labels[bestFeat]
    # myTree = {bestFeatLabel: {}}
    # del (labels[bestFeat])
    # featValues = [example[bestFeat] for example in dataset]
    # uniqueVals = set(featValues)
    # for value in uniqueVals:
    #     subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
    #     myTree[bestFeatLabel][value] = create_tree(split_dataset(dataset, bestFeat, value), subLabels)
    # return myTree


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


def store_tree(input_tree, filename='classifier_storage.txt'):
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename='classifier_storage.txt'):
    fr = open(filename)
    return pickle.load(fr)
