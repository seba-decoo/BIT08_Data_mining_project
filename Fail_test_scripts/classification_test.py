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
#https://www.datacamp.com/community/tutorials/decision-tree-classification-python 
####################################################################################################

#import packages
from numpy import clongfloat
import pandas
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split

#load data
ds = pandas.read_csv("heart_preprocessed.csv",)

##construct classification tree

'''
#encode categorial values when needed
from category_encoders import OrdinalEncoder
# Categorical boolean mask
categorical_feature_mask = ds.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = ds.columns[categorical_feature_mask].tolist()
ds= OrdinalEncoder(cols=categorical_cols).fit_transform(ds)

'''
#define X and y value
X = ds.drop('target', axis=1)
y = ds['target']

#split data in training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) # specify size of training set ##add random_state =1

#build up decision tree
#plt.figure(figsize=[14,12])     #uncomment for direct visualisation
clf = DecisionTreeClassifier(random_state=0, max_depth=4, criterion="entropy")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#tree.plot_tree(decision_tree)
#plt.show()     #uncomment for direct visualisation

#visualize with graphviz 
#!!! add to path of OS (explanation --> see begin of script)!!!

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names= X.columns ,class_names=['less chance','more chance'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_heart.png')
Image(graph.create_png())

#evaluation criteria for the decision tree 
from sklearn.metrics import classification_report, confusion_matrix
print("Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")
print("QUALITY REPORT:")
print(classification_report(y_test, y_pred))

