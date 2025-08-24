# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 01:27:15 2021

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

train_dataset = pd.read_csv("training-set.csv")
test_dataset = pd.read_csv("testing-set.csv")

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
x_train = dataset.iloc[:175300, 0:-1].values
y_train = dataset.iloc[:175300, -1].values
x_test = dataset.iloc[35060:, 0:-1].values
y_test = dataset.iloc[35060:, -1].values


clf1 = MLPClassifier(solver='adam',
                      hidden_layer_sizes=(5, 5),
                      activation='relu',
                      max_iter = 10,
                      verbose=True,
                      tol= 1e-100,
                      n_iter_no_change=1,
                      early_stopping=True,
                      warm_start=True)


#clf2 =  RandomForestClassifier(random_state=1)
#estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

parameter_gridsearch = {
'max_depth' : [1, 2],  #depth of each decision tree
'n_estimators': [2, 1],  #count of decision tree
'max_features': ['sqrt', 'auto', 'log2'],
'min_samples_split': [2],
'min_samples_leaf': [1, 3, 4],
'bootstrap': [True, False],
}

clf2 = RandomForestClassifier()
clf2 = GridSearchCV(clf2,             #grid search for algorithm optimization
scoring='accuracy',
param_grid=parameter_gridsearch,
cv=10)


lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1 , clf2], 
                          meta_classifier=lr)
print('3-fold cross validation:\n')
for clf, label in zip([ clf1, clf2, sclf], 
                      ['Multi Layers Perceptron', #from sklearn.neural_network import MLPClassifier
                       'Random Forest',
                       'Stacking Classifier']):
    history = clf.fit(x_train, y_train)

    scores = model_selection.cross_val_score(clf, x_train, y_train, 
                                              cv=50, scoring='accuracy') # cv = EPOCH
    scores_test = model_selection.cross_val_score(clf, x_test, y_test, 
                                              cv=50, scoring='accuracy')
    predictions = clf.predict(x_test[:400])
    target  = y_test[:400]
    train_y = y_train[:400]
    correct = 0
    wrong = 0
    for i in range( len(predictions)):
        if (predictions[i] == target[i]):
            correct += 1
        else:
            wrong += 1
        print(predictions[i] , target[i])

    print('correct:', correct)
    print('wrong:', wrong)



    acc = accuracy_score(target, predictions)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (acc, scores.std(), label))
  

    plt.plot(scores, color='blue',
        marker='.', linestyle='dashed', linewidth=2, markersize=12)
    plt.plot(scores_test, color='red',
      marker='.', linestyle='dashdot', linewidth=2, markersize=12)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    lasso = linear_model.Lasso()
    losss = cross_val_score(lasso,  x_train, y_train, cv=50)
    losss_test = cross_val_score(lasso,  x_test, y_test, cv=50)
    plt.plot(losss, color='blue',
            marker='.', linestyle='dashed', linewidth=2, markersize=12)
    plt.plot(losss_test, color='red',
        marker='.', linestyle='dotted', linewidth=2, markersize=12)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()