from numpy import *
import operator
from os import listdir


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, data_set, labels, k=2):
    data_size = data_set.shape[0]
    diff_mat = tile(inX, (data_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sortedDistances = argsort(distances)
    class_count = {}
    for _ in range(k):
        vote_label = labels[sortedDistances[_]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    return sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]


def file2matrix(filename):
    fr = open(filename)
    array_lines = fr.readlines()
    return_mat = zeros((len(array_lines), 3))
    class_label_vector = []
    for index in range(len(array_lines)):
        line = array_lines[index]
        line = line.strip().split('\t')
        return_mat[index] = line[:3]
        class_label_vector.append(line[-1])
    return return_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    norm_data_set = (data_set - tile(min_vals, (data_set.shape[0], 1))) / (max_vals - min_vals)
    return norm_data_set


def dating_class_test():
    ho_ratio = 0.90
    dating_data_mat, dating_labels = file2matrix('data/datingTestSet2.txt')
    norm_mat = auto_norm(dating_data_mat)
    num_test_vecs = int(norm_mat.shape[0] * ho_ratio)
    error_count = 0
    for index in range(num_test_vecs):
        classify = classify0(dating_data_mat[index], dating_data_mat, dating_labels, 2)
        if classify != dating_labels[index]:
            error_count += 1
        print 'the classifier came back with: %s, the read answer is: %s' % (classify, dating_labels[index])
    print 'the total error rate is: %f' % (float(error_count) / float(num_test_vecs))


def img2vector(filename):
    return_vector = zeros((1, 1024))
    fr = open(filename)
    file_str = fr.readlines()
    for i in range(len(file_str)):
        line_str = file_str[i].strip()
        for j in range(len(line_str)):
            return_vector[0, 32 * i + j] = int(line_str[j])
    return return_vector


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('data/trainingDigits')
    m = len(training_file_list)
    training_mat = zeros((m, 32 * 32))
    for index in range(m):
        file_name_str = training_file_list[index]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[index, :] = img2vector('data/trainingDigits/%s' % file_name_str)
    test_file_list = listdir('testDigits')
    error_count = 0
    m_test = len(test_file_list)
    for index in range(len(test_file_list)):
        file_name_str = test_file_list[index]
        class_name_str = file_name_str.split('.')[0].split('_')[0]
        vector_under_test = img2vector('data/testDigits/%s' % test_file_list[index])
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 2)
        print 'the classifier came back with: %s, the read answer is: %s' % (classifier_result, class_name_str)
        if int(class_name_str) == int(classifier_result):
            error_count += 1
    print 'the total error rate is: %f' % (float(20) / float(m_test))


handwriting_class_test()
