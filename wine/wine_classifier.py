#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from scipy import stats

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']    

def overlapping_numbers(feature, labels):
    olp_ns = np.zeros(3, dtype=int)
    for i in range(3):
        index = np.concatenate(np.argwhere(labels == i+1), axis=0)
        group_max = max(np.take(feature, index))
        group_min = min(np.take(feature, index))
        for j in range(feature.shape[0]):
            if feature[j] > group_min and feature[j] < group_max and labels[j] != i+1:
                olp_ns[i] += 1
    return olp_ns

def feature_selection(train_set, train_labels, n = 2, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function
    features = []
    measurements = []
    for i in range(train_set.shape[1]):
        olp = overlapping_numbers(train_set[:,i], train_labels)
        measurements.append(sum(olp))
    for i in range(n):
        #argmin only takes the first of min
        features.append(np.argmin(measurements))
        measurements[np.argmin(measurements)] = float('inf')
    return features    


def knn(train_set, train_labels, test_set, k, features = [6,9], **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    # features = feature_selection(train_set, train_labels)
    train_set_2d = train_set[:, features]
    test_set_2d = test_set[:, features]

    predictions = np.zeros(test_set.shape[0], dtype=int)

    for test_index in range(test_set.shape[0]): 
        test_feature1 = test_set_2d[test_index, 0]
        test_feature2 = test_set_2d[test_index, 1]
        distance = np.zeros(train_set.shape[0])

        for train_index in range (train_labels.shape[0]):
            train_feature1 = train_set_2d[train_index, 0]
            train_feature2 = train_set_2d[train_index, 1]
            distance[train_index] = ((test_feature1 - train_feature1)**2 + (test_feature2 - train_feature2)**2)**0.5

        label_counter = np.zeros(3)
        for i in range(k): 
            label = train_labels[np.argmin(distance)]
            label_counter[label-1] += 1
            distance[np.argmin(distance)] = float('inf')
        
        predictions[test_index] = np.argmax(label_counter) + 1

    return predictions


# def knn(train_set, train_labels, test_set, k, n = 2, **kwargs):
#     # write your code here and make sure you return the predictions at the end of 
#     # the function
#     features = feature_selection(train_set, train_labels)
#     train_set_nd = train_set[:, features]
#     test_set_nd = test_set[:, features]

#     predictions = np.zeros(test_set.shape[0], dtype=int)
    
#     for test_index in range(test_set.shape[0]): 
#         for i in range(n): 
#             test_feature = np.zeros(n)
#             test_feature[i] = test_set_nd[test_index, i]

#         distance = np.zeros(train_set.shape[0])
#         for train_index in range (train_labels.shape[0]):
#             for i in range(n): 
#                 train_feature = np.zeros(n)
#                 train_feature[i] = train_set_nd[train_index, i]

#             distance[train_index] = np.sum(np.square(test_feature - train_feature)) ** 0.5

#         label_counter = np.zeros(3)
#         for i in range(k): 
#             label = train_labels[np.argmin(distance)]
#             label_counter[label-1] += 1
#             distance[np.argmin(distance)] = float('inf')

#         predictions[test_index] = np.argmax(label_counter) + 1

#     return predictions

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    features = feature_selection(train_set, train_labels)
    train_set_2d = train_set[:, features]
    test_set_2d = test_set[:, features]

    predictions = np.zeros(test_set.shape[0], dtype=int)

    indices_of_label1 = np.argwhere(train_labels == 1).flatten('F')
    indices_of_label2 = np.argwhere(train_labels == 2).flatten('F')
    indices_of_label3 = np.argwhere(train_labels == 3).flatten('F')

    train_set_2d_label1 = train_set_2d[indices_of_label1, :]
    train_set_2d_label2 = train_set_2d[indices_of_label2, :]
    train_set_2d_label3 = train_set_2d[indices_of_label3, :]
    
    cm1 = np.cov(train_set_2d_label1.T)
    mu1 = np.mean(train_set_2d_label1, axis = 0)
    cm2 = np.cov(train_set_2d_label2.T)
    mu2 = np.mean(train_set_2d_label2, axis = 0)
    cm3 = np.cov(train_set_2d_label3.T)
    mu3 = np.mean(train_set_2d_label3, axis = 0)
    
    for i in range(test_set.shape[0]):
        p1 = stats.multivariate_normal.pdf(test_set_2d[i,:], mean = mu1, cov = cm1)
        p2 = stats.multivariate_normal.pdf(test_set_2d[i,:], mean = mu2, cov = cm2)
        p3 = stats.multivariate_normal.pdf(test_set_2d[i,:], mean = mu3, cov = cm3)
        predictions[i] = np.argmax([p1, p2, p3])+1

    return predictions


def knn_three_features(train_set, train_labels, test_set, k, features = [0, 6, 12],**kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    train_set_3d = train_set[:, features]
    test_set_3d = test_set[:, features]

    predictions = np.zeros(test_set.shape[0], dtype=int)

    for test_index in range(test_set.shape[0]): 
        test_feature1 = test_set_3d[test_index, 0]
        test_feature2 = test_set_3d[test_index, 1]
        test_feature3 = test_set_3d[test_index, 2]
        distance = np.zeros(train_set.shape[0])

        for train_index in range (train_labels.shape[0]):
            train_feature1 = train_set_3d[train_index, 0]
            train_feature2 = train_set_3d[train_index, 1]
            train_feature3 = train_set_3d[train_index, 2]
            distance[train_index] = ((test_feature1 - train_feature1)**2 + 
                                        (test_feature2 - train_feature2)**2 +
                                        (test_feature3 - train_feature3)**2) ** 0.5

        label_counter = np.zeros(3)
        for i in range(k): 
            label = train_labels[np.argmin(distance)]
            label_counter[label-1] += 1
            distance[np.argmin(distance)] = float('inf')
        
        predictions[test_index] = np.argmax(label_counter) + 1

    return predictions


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        # Print accuracy along with result
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
        correct_count = float(0)
        for m in range(test_labels.shape[0]):
            if predictions[m] == test_labels[m]:
                correct_count +=1
        accuracy = correct_count / test_labels.shape[0]
        print(accuracy)

        # # Print a sorted array wrt accuracy of all feature pair combinations
        # for k in range(1,8):
        #     accuracy_rankings =  np.zeros((156, 3))
        #     index = 0
        #     for i in range(0, 13):
        #         for j in range(0, 13):
        #             if i != j: 
        #                 predictions = knn(train_set, train_labels, test_set, k, features = [i, j])
        #                 #print_predictions(predictions)
        #                 correct_count = float(0)
        #                 for m in range(test_labels.shape[0]):
        #                     if predictions[m] == test_labels[m]:
        #                         correct_count +=1
        #                 accuracy=correct_count/test_labels.shape[0]
        #                 accuracy_rankings[index, 0] = i
        #                 accuracy_rankings[index, 1] = j
        #                 accuracy_rankings[index, 2] = accuracy
        #                 index += 1
        #     sort =  np.argsort(accuracy_rankings, axis = 0)
        #     print(k)
        #     print(accuracy_rankings[sort[146:155,2], :])

        # # Print the accuracy of a given pair of features using different k
        # accuracy_rankings = np.zeros((7,2))
        # for k in range(1,8):
        #     predictions = knn(train_set, train_labels, test_set, k)
        #     #print_predictions(predictions)
        #     correct_count = float(0)
        #     for m in range(test_labels.shape[0]):
        #         if predictions[m] == test_labels[m]:
        #             correct_count +=1
        #     accuracy = correct_count / test_labels.shape[0]
        #     accuracy_rankings[k-1,0] = k
        #     accuracy_rankings[k-1,1] = accuracy
        # print(accuracy_rankings)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        # predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        # print_predictions(predictions)
        for j in [0,1,2,3,4,5,7,8,10,11,12]:
            features = [j, 6, 9]
            predictions = knn_three_features(train_set, train_labels, test_set, args.k, features)
            print_predictions(predictions)
            correct_count = float(0)
            for i in range(test_labels.shape[0]):
                if predictions[i] == test_labels[i]:
                    correct_count +=1
            print(correct_count/test_labels.shape[0])
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))