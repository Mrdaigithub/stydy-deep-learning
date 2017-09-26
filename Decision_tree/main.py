# coding: utf-8
from numpy import *
from math import log
import trees
import pickle


def get_lenses_data(filename):
    fr = open(filename)
    source_data = fr.readlines()
    _lenses_data = [i.strip().split('\t') for i in source_data]
    _lenses_labels = ['age', 'spectacle_prescription', 'astigmatic', 'tear_production_rate']
    return _lenses_data, _lenses_labels


if __name__ == '__main__':
    lenses_dataset, lenses_labels = get_lenses_data('./data/lenses.txt')
    # lenses_dataset, lenses_labels = trees.create_dataset()
    print trees.create_tree(lenses_dataset, lenses_labels)
