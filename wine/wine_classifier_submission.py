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
# My code
# Blue, Red, Yellow
class_colours = np.array([CLASS_1_C, CLASS_2_C, CLASS_3_C])
train_set, train_labels, test_set, test_labels = load_data()   

def myFunction_graph_combinations(train_set, train_labels):
    n_features = train_set.shape[1]
    fig, ax = plt.subplots(n_features, n_features)
    plt.subplots_adjust(left=0.01, right=5.99, top=5.99, bottom=0.01, wspace=0.2, hspace=0.4)
    for i in range(n_features):
        for j in range(n_features):
            ax[i][j].scatter(train_set[:, i], train_set[:, j], c=class_colours[train_labels[:]-1])
            ax[i][j].set_title('Features {} vs {}'.format(i+1, j+1))

# 1-B 2-R 3-Y
def overlapping_numbers(feature, labels):
    olp_ns = np.zeros(6, dtype=int)
    for i in range(3):
        index = np.concatenate(np.argwhere(labels == i+1), axis=0)
        group_max = max(np.take(feature, index))
        group_min = min(np.take(feature, index))
        for j in range(feature.shape[0]):
            if feature[j] > group_min and feature[j] < group_max and labels[j] == (i+1)%3+1:
                olp_ns[2*i] += 1
            if feature[j] > group_min and feature[j] < group_max and labels[j] == (i+2)%3+1:
                olp_ns[2*i+1] += 1
    return olp_ns

def combinations(index, d, list_f):
    global temp_comb
    global min_comb
    global min_dotp
    if d == 1:
        for i in range(index, len(list_f)):
            temp_comb.append(i)
            dotp = 0
            for j in range(list_f[0].shape[0]):
                product = 1
                for k in temp_comb:
                    product *= list_f[k][j]
                dotp += product
            if dotp < min_dotp:
                min_dotp = dotp
                for m in range(len(temp_comb)):
                    min_comb[m] = temp_comb[m]
            del temp_comb[-1]
    else: 
        if d > 1:
            for i in range(index, len(list_f)-d+1):
                temp_comb.append(i)
                combinations(i+1, d-1, list_f)
                del temp_comb[-1]

def feature_selection(train_set, train_labels, f=2, **kwargs):
    # write your code here and make sure you return the features at the end of 
    # the function
    measurements = []
    for i in range(train_set.shape[1]):
        olp = overlapping_numbers(train_set[:,i], train_labels)
        measurements.append(olp)
    global min_dotp 
    min_dotp = float('inf')
    global min_comb
    min_comb = []
    for i in range(f):
        min_comb.append(i)
    global temp_comb
    temp_comb = []
    combinations(0, f, measurements)
    result = []
    for i in range(len(min_comb)):
        result.append(min_comb[i])
    del min_comb[-f:]
    return result

def knn_core(train_set_f, train_labels, test_set_f, k):
    predict_labels = np.zeros(test_set_f.shape[0], dtype='int')
    
    for t in range(test_set_f.shape[0]):
        distances = np.zeros(train_set_f.shape[0])
        for i in range(train_set_f.shape[0]):
            for j in range(train_set_f.shape[1]):
                distances[i] += (test_set_f[t][j] - train_set_f[i][j]) ** 2
            distances[i] = distances[i] ** 0.5
        class_counter = np.zeros(3)
        for i in range(k):
            class_counter[train_labels[np.argmin(distances)] - 1] += 1
            distances[np.argmin(distances)] = float('inf')
        predict_labels[t] = np.argmax(class_counter) + 1
    return predict_labels

def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    features = feature_selection(train_set, train_labels)
    train_set_f = train_set[:, features]
    test_set_f = test_set[:, features]
    return knn_core(train_set_f, train_labels, test_set_f, k)

def confusion_matrix(predictions, test_labels):
    matrix = np.zeros((3,3))
    for i in range(3):
        class_i = np.where(test_labels == i+1)[0]
        for j in range(3):
            matrix[i][j] = len(np.where(predictions[class_i] == j+1)[0]) / len(class_i)
    return matrix

def plot_matrix(matrix, title=None, xlabel=None, ylabel=None, ax=None):
    if ax is None:
        ax = plt.gca()
    
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap=plt.get_cmap('summer'))
    ax.set_xticks(np.arange(matrix.shape[0]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    fig.colorbar(im)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

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

def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    features = [0, 6, 9]
    train_set_f = train_set[:, features]
    test_set_f = test_set[:, features]
    predictions = knn_core(train_set_f, train_labels, test_set_f, k)
    return predictions

def pca_core(data_set, n_components):
    norm_data_set = data_set - np.mean(data_set, axis = 0)
    cov_matrix = np.cov(norm_data_set, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    order = np.flip(np.argsort(eigenvalues), axis = 0)[0:n_components]
    W = eigenvectors[:,order]
    myPCA = np.dot(norm_data_set, W)
    return myPCA

def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function
    pca_train_set = pca_core(train_set, n_components)
    pca_test_set = pca_core(test_set, n_components)
    pred_pca = knn_core(pca_train_set, train_labels, pca_test_set, k)
    pca_plot = plt.subplot()
    pca_plot.scatter(pca_train_set[:, 0], pca_train_set[:, 1], c=class_colours[train_labels[:]-1])
    pca_plot.set_title('PCA plot')
    return pred_pca

def accuracy(pred, label):
    counter = float(0)
    for i in range(pred.shape[0]):
        if pred[i] == label[i]:
            counter += 1
    return counter/len(pred)

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
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))