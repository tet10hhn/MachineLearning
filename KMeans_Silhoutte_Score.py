#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 22:36:32 2021

@author: hassan
"""


# scatter plot of blobs dataset
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from numpy import where
# generate 2d classification dataset
X, y = make_blobs(n_samples=1000, centers=20, n_features=100, cluster_std=1, random_state=10)
# scatter plot for each class value
for class_value in range(3):
    
# select indices of points with the class label
    row_ix = where(y == class_value)
    # scatter plot for points with a different color
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
pyplot.show()
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, random_state=1).fit(X)
target_predicted = model.labels_
silhouette_score(X, target_predicted)