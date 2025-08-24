# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 01:29:35 2021

@author: Mahsa
"""

import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
#from sklearn.cross_validation import StratifiedKFold # Add important libs)

train_dataset = pd.read_csv("train.csv")
test_dataset = pd.read_csv("test.csv")
train_dataset = train_dataset.drop(
    train_dataset.columns[[0, 2, 3, 4, 43]], axis=1)
test_dataset = test_dataset.drop(
    test_dataset.columns[[0, 2, 3, 4, 43]], axis=1)
frames = [train_dataset, test_dataset]
dataset = pd.concat(frames)
dataset = dataset.sample(frac=1)
dataset = dataset.astype('float32')
#dataset = np.concatenate((train_dataset, test_dataset), axis=0)
# np.random.shuffle(dataset)
x_train = dataset.iloc[:100, 0:-1].values
y_train = dataset.iloc[:100, -1].values
x_test = dataset.iloc[20:, 0:-1].values
y_test = dataset.iloc[20:, -1].values
parameter_gridsearch = {
'max_depth' : [3, 4],  #depth of each decision tree
'n_estimators': [5, 2],  #count of decision tree
'max_features': ['sqrt', 'auto', 'log2'],
'min_samples_split': [2],
'min_samples_leaf': [1, 3, 4],
'bootstrap': [True, False],
}

randomforest = RandomForestClassifier()
#crossvalidation = StratifiedKFold(y_train , n_folds=5)

gridsearch = GridSearchCV(randomforest,             #grid search for algorithm optimization
scoring='accuracy',
param_grid=parameter_gridsearch,
cv=10)
gridsearch.fit(x_train, y_train)    #train[0::,0] is as target
clf1 = gridsearch
parameters = clf1.best_params_

scores = model_selection.cross_val_score(clf1, x_train, y_train, 
                                          cv=10, scoring='accuracy') # cv = EPOCH
scores_test = model_selection.cross_val_score(clf1, x_test, y_test, 
                                          cv=10, scoring='accuracy')
    
plt.plot(scores, color='blue',
    marker='.', linestyle='dotted', linewidth=2, markersize=12)
plt.plot(scores_test, color='red',
  marker='.', linestyle='dashdot', linewidth=2, markersize=12)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print(scores)
print(scores_test)
