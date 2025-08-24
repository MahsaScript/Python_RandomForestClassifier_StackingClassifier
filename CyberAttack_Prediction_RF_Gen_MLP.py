#####!pip install sklearn-genetic
import numpy as np
from sklearn import datasets, linear_model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.neural_network import MLPClassifier
import numpy as np
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
x_train = dataset.iloc[:257000, 0:-1].values
y_train = dataset.iloc[:257000, -1].values
x_test = dataset.iloc[25001:, 0:-1].values
y_test = dataset.iloc[25001:, -1].values

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
clf1 = MLPClassifier(solver='adam',
                      hidden_layer_sizes=(5, 5),
                      activation='relu',
                      max_iter = 350,
                      verbose=True,
                      tol= 1e-100,
                      n_iter_no_change=10,
                      early_stopping=True,
                      warm_start=True)
#epoch = max_iter and cv from below

randomforest  = RandomForestClassifier()

parameter_gridsearch = {
'max_depth' : [3, 4],  #depth of each decision tree
'n_estimators': [50, 20],  #count of decision tree
'max_features': ['sqrt', 'auto', 'log2'],
'min_samples_split': [2],
'min_samples_leaf': [1, 3, 4],
'bootstrap': [True, False],
}

clf2 = GridSearchCV(randomforest,             #grid search for algorithm optimization
    scoring='accuracy',
    param_grid=parameter_gridsearch,
    cv=350)

estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
clf3 = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=2,
                              n_population=10,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=10,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=4,
                              caching=True,
                              n_jobs=-1)

#http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1 , clf2 , clf3], 
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([ clf1, clf2, clf3, sclf], 
                      ['Multi Layers Perceptron', #from sklearn.neural_network import MLPClassifier
                       'Random Forest',
                       'Genetic Algorithm',
                       'StackingClassifier']):
                       
    history = clf.fit(x_train, y_train)
    scores = model_selection.cross_val_score(clf, x_train, y_train, 
                                              cv=350, scoring='accuracy') # cv =EPOCH
    
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    predictions = clf.predict(x_test[:100])
    target = y_test[:100]
    correct = 0
    wrong = 0
    for i in range( len(predictions)):
        if (predictions[i] == target[i]):
            correct += 1
        else:
            wrong += 1
        print(predictions[i] , target[i])
    plt.plot(scores, color='blue',
         marker='.', linestyle='dashed', linewidth=2, markersize=12)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    print('correct:', correct)
    print('wrong:', wrong)