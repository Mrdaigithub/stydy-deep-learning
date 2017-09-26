import trees


if __name__ == '__main__':
    lenses_dataset, lenses_labels = trees.get_training_lenses_data('./data/lenses.txt')
    tree = trees.create_tree(lenses_dataset, lenses_labels)
    trees.store_tree(tree, './data/lenses_tree.txt')
