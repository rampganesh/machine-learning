# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 18:38:01 2018

@author: julio47

Install graphviz and pydot (In Linux)

Plots taken from 'https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Chapter%208.ipynb'
"""

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz;
from sklearn.model_selection import train_test_split, GridSearchCV;
from sklearn.metrics import mean_squared_error,confusion_matrix, classification_report;
import pydot
from IPython.display import Image;
from sklearn.externals.six import StringIO;
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor;


def print_tree(estimator, features, class_names=None, filled=True):
#    tree = estimator
#    names = features
#    color = filled
    classn = class_names
    
    dot_data = StringIO()
    string = export_graphviz(estimator, out_file = dot_data, feature_names=features, class_names=classn, filled=filled)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return(graph)

dataset = pd.read_csv('Carseats.csv')

dataset['High'] = np.where(dataset["Sales"] <= 8, "No", "Yes")

dataset.Urban = dataset.Urban.map({'No':0, 'Yes':1})

dataset.US = dataset.US.map({'No':0, 'Yes':1})

dataset.ShelveLoc = pd.factorize(dataset.ShelveLoc)[0]

# Predictor and response

X = dataset.drop(['Sales', 'High'], axis = 1)

y = dataset['High']

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.5)


carseats_tree = DecisionTreeClassifier()

carseats_tree.fit(X_train, y_train)

print(classification_report(y_train, carseats_tree.predict(X_train)))

# Visualising the tree

graph, = print_tree(carseats_tree, features=X.columns, class_names=['No', 'Yes'])
Image(graph.create_png())

# Testing the tree

y_pred = carseats_tree.predict(X_test)

confusion_matrix(y_test, y_pred)

# 46/200 = 27% error rate

# Selecting the best depth using grid cross validation
"""
tree_classifier = DecisionTreeClassifier()

leaf_range = list(range(2,31))

depth_range = list(range(2, 10))

sample_leaf_range = list(range(1,5))

param_grid = dict(max_leaf_nodes = leaf_range, max_depth = depth_range, min_samples_leaf = sample_leaf_range)

print(param_grid)

grid = GridSearchCV(tree_classifier, param_grid, cv = 10, scoring = "accuracy")

grid.fit(X, y)

grid.grid_scores_

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

np.max(grid_mean_scores)

"""

################## Regression Tree #####################

dataset = pd.read_csv('Boston.csv', index_col = 0)

X = dataset.drop(["medv"], axis = 1)

y = dataset['medv']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = 0.5)



regressor = DecisionTreeRegressor(random_state = 1)

regressor.fit(X_train, y_train)

preds = regressor.predict(X_test)

mean_squared_error(y_test, preds)


# Visualising the tree

graph, = print_tree(regressor, features=X.columns)
Image(graph.create_png())


plt.scatter(preds, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')



############################# Random Forest Regression #############################


regressor = RandomForestRegressor(max_features=13, random_state=1, oob_score=True)

regressor.fit(X_train, y_train)

regressor.oob_score_

preds = regressor.predict(X_test)

plt.scatter(preds, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y_test, preds)

regressor.feature_importances_

Importance = pd.DataFrame({'Importance':regressor.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


# Reducing the max_features

regressor = RandomForestRegressor(max_features=6, random_state=1, oob_score=True)

regressor.fit(X_train, y_train)

regressor.oob_score_

preds = regressor.predict(X_test)

plt.scatter(preds, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('pred')
plt.ylabel('y_test')

mean_squared_error(y_test, preds)

regressor.feature_importances_

Importance = pd.DataFrame({'Importance':regressor.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None


####################### Boosting ############################


regressor = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.01, random_state=1)

regressor.fit(X_train, y_train)

feature_importance = regressor.feature_importances_*100
rel_imp = pd.Series(feature_importance, index=X.columns).sort_values(inplace=False)
print(rel_imp)
rel_imp.T.plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None

preds = regressor.predict(X_test)

mean_squared_error(y_test, preds)

