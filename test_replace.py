#test replace function on dataset


from os import sep
import pandas

ds = pandas.read_csv("heart.csv")

print(ds)

ds["sex"].replace({0:"F", 1:"M"}, inplace=True)
ds["cp"].replace({0:"typ_ang", 1:"atyp_ang", 2:"non-ang_pain", 3:"asymp"}, inplace= True)
ds["fbs"].replace({1:True, 0:False}, inplace=True)
ds["restecg"].replace({0:"normal", 1:"ST-T", 2:"LV_hyper"}, inplace= True)
ds["exang"].replace({0:"no", 1:"yes"}, inplace=True)
ds["slope"].replace({0:"UP", 1:"FL", 2:"DW" }, inplace= True)
ds["thal"].replace({1:"normal", 2:"fix_defect", 3:"revers_defect" }, inplace= True)
ds["target"].replace({0:"less_change", 1:"more_change"}, inplace=True)


print(ds)

ds.to_csv("heart_preprocessed.csv", sep=",", index=False)

##construct classification tree
#define X and y value
X = ds.drop('target', axis=1)

#encode categorial values

from category_encoders import OrdinalEncoder

# Categorical boolean mask
categorical_feature_mask = ds.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = ds.columns[categorical_feature_mask].tolist()

X= OrdinalEncoder(cols=categorical_cols).fit_transform(ds)

#print(X)


y = ds['target']
#split data in training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, \
    test_size=0.33) # specify size of training set

#build up decision tree

#plt.figure(figsize=[14,12])     #uncomment for direct visualisation
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(X_train, y_train)
tree.plot_tree(decision_tree)
#plt.show()     #uncomment for direct visualisation

#visualize with graphviz 
#!!! add to path of OS (explanation --> see begin of script)!!!

import graphviz
dot_data = tree.export_graphviz(decision_tree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("dicision_tree")

#visualize text_based tree
attr_names = list(ds.columns)
attr_names.pop(-1)
r = export_text(decision_tree, attr_names)
print(r)

#evaluation
y_pred = decision_tree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''
##################################################################
#encode categorie attributes
#link to expanation
#https://scikit-learn.org/stable/modules/preprocessing.html (6.3.4)
#https://towardsdatascience.com/encoding-categorical-features-21a2651a065c

# Categorical boolean mask
categorical_feature_mask = ds.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = ds.columns[categorical_feature_mask].tolist()
categorical_features = ds.select_dtypes("object").columns

print(categorical_feature_mask)
print(categorical_cols)



# import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# instantiate OneHotEncoder
ohe = OneHotEncoder(categories=categorical_feature_mask, sparse=False ) 
# categorical_features = boolean mask for categorical columns
# sparse = False output an array not sparse matrix
# apply OneHotEncoder on categorical feature columns
X_ohe = ohe.fit_transform(ds) # It returns an numpy array
'''

