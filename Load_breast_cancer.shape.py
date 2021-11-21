# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:18:48 2020

@author: Hassan
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)
print(X_train.shape)
print(X_test.shape)
