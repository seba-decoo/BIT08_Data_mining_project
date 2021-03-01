####################################################################################################
##CLASSIFICATION
####################################################################################################
#https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

#import packages
import pandas
from sklearn import tree
import sklearn
#load data

ds = pandas.read_csv("heart.csv",)

#construct classification tree

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

X = ds.drop('target', axis=1)
y = ds['target']

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

decision_tree = DecisionTreeClassifier(random_state=1234)
decision_tree = decision_tree.fit(X, y)

tree.export_text(decision_tree)
'''
r = export_text(decision_tree, feature_names= X.columns)
print(r)


y_pred = decision_tree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import graphviz 
dot_data = tree.export_graphviz(decision_tree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 
'''