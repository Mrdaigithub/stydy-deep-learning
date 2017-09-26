import trees


def classify(input_tree, feature_labels, test_vec):
    if type(input_tree).__name__ != 'dict':
        return input_tree
    key = input_tree.keys()[0]
    label_index = feature_labels.index(key)
    sub_input_tree = input_tree[key][test_vec[label_index]]
    return classify(sub_input_tree, feature_labels, test_vec)


if __name__ == '__main__':
    lenses_dataset, lenses_labels = trees.get_training_lenses_data('./data/lenses.txt')
    tree = trees.fetch_tree('./data/lenses_tree.txt')
    print classify(tree, lenses_labels, ['presbyopic', 'myope', 'no', 'normal'])
