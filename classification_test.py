####################################################################################################
#CLASSIFICATION
####################################################################################################
# FOR WINDOWS:  install graphviz via this link: https://graphviz.org/download/ 
#               and install package with pip install graphviz
#               !!! reload all opened terminals !!!

# FOR LINUX:    pip3 install graphviz
####################################################################################################
##links to explanation
#https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
####################################################################################################

#import packages
import pandas
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split

#load data
ds = pandas.read_csv("heart_preprocessed.csv",)

##construct classification tree
#define X and y value


#encode categorial values

from category_encoders import OrdinalEncoder

# Categorical boolean mask
categorical_feature_mask = ds.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = ds.columns[categorical_feature_mask].tolist()

ds= OrdinalEncoder(cols=categorical_cols).fit_transform(ds)

print(ds)
X = ds.drop('target', axis=1)
y = ds['target']


#split data in training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, \
    test_size=0.33) # specify size of training set

#build up decision tree

plt.figure(figsize=[14,12])     #uncomment for direct visualisation
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X_train, y_train)
tree.plot_tree(decision_tree)
plt.show()     #uncomment for direct visualisation

#visualize with graphviz 
#!!! add to path of OS (explanation --> see begin of script)!!!

import graphviz
dot_data = tree.export_graphviz(decision_tree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("dicision_tree_enctest")

#visualize text_based tree
attr_names = list(ds.columns)
attr_names.pop(-1)
r = export_text(decision_tree, attr_names)
print(r)
'''
#evaluation
y_pred = decision_tree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''