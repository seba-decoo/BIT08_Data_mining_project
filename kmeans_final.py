####################################################################################################
####################################################################################################
#KMEANS CLUSTERING (descriptive data mining)
####################################################################################################
####################################################################################################

##Import modules
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas
import sklearn.cluster as cluster
from sklearn.metrics import silhouette_score

##Reading in data from CSV
df = pandas.read_csv("heart.csv")

##Removing the class attribute for unsupervised clustering
dfnt = df.drop(columns=["target"])

####################################################################################################
#Determine amount of clusters
####################################################################################################
print("DETERMINATING THE AMOUNT OF CLUSTERS\n...\n...")

##Silhouette coefficient
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
##List for silhouette_coefficient and library for saving with amount of clusters
silhouette_coefficients = []
clust_lib = {}

##Loop over amount of clusters (2-11)
for k in range(2, 11):
    kmeans = cluster.KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(dfnt)
    score = silhouette_score(dfnt, kmeans.labels_)
    silhouette_coefficients.append(score)
    clust_lib[score] = k

##Plot silhouette coefficients
sil = plt.figure(figsize=[10,5])
#plt.style.use("fivethirtyeight")   ### Applying stylesheet causes all plots to use the stylesheet
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
#plt.show() #show script during executing
plt.savefig('Images/silhouette_coefficient.png') #save figure in directory 

#Determine nr of clusters for kmeans clustering
nr_clusters= clust_lib[max(silhouette_coefficients)]
print("Amount of cluster for kmeans clustering:\t{}\nThe silhouette coëfficient:\t{}".format(nr_clusters, max(silhouette_coefficients)))

##Stop script for 3 seconds
import time
x=0 
while x <= 2:   ###fixed loop
    time.sleep(1)
    print("...")
    x+=1

####################################################################################################
#Clustering
####################################################################################################
print("CLUSTERING STARTED\n...\n...")

##Execution of clustering
kmeans = cluster.KMeans(n_clusters=nr_clusters).fit(dfnt)
y = kmeans.predict(dfnt)

##Script to go over all attributes and plot them against each other. try to do this for kmeans. can be compared to weka
#print(dfnt)

spec2 = GridSpec(ncols=13, nrows=13, wspace=0, hspace=0)
fig = plt.figure(figsize=[16,9])

a=0
for att in dfnt:
    for a2 in dfnt:
        ax = fig.add_subplot(spec2[a])
        plt.scatter(dfnt[att],dfnt[a2],c=y, cmap="prism", s=5, marker=".")
        plt.yticks([])
        plt.xticks([])

        # label y 
        if ax.is_first_col():
            ax.set_ylabel(att, fontsize = 9)

        # label x 
        if ax.is_first_row():
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(a2, fontsize = 9)
        a += 1

#plt.show()
plt.savefig('Images/kmeans.png') #save figure in directory 
#Instructions to results
print("CLUSTERING TERMINATED\n...\n...")
print("The silhouette plot and the clustering output are saved in the Images folder.\n")

####################################################################################################
#Clustering prediction
####################################################################################################

print("CLUSTERING EVALUATION:\n")

#Calculate procent (in)correct
y_in = 0
actual = (df["target"])
correct = 0
incorrect = 0

correct_0 = 0
correct_1 = 0
incorrect_0 = 0
incorrect_1 = 0

for act in actual:
    if act == y[y_in]:
        correct += 1
        if act == 0:
            correct_0 += 1
        else:
            correct_1 += 1
    else:
        incorrect += 1
        if act == 0:
            incorrect_0 += 1
        else:
            incorrect_1 += 1
    y_in += 1

#Print procent (in)correct
print("Procent correct: \t{:.2f}%".format(correct/3.03))
print("Procent incorrect: \t{:.2f}%\n".format(incorrect/3.03))

##Constructing confusion matrix
print("\t0\'\t1\'")
print("="*50)
print("\t{}\t{}\t 0 = decreased chance".format(correct_0,incorrect_0))
print("\t{}\t{}\t 1 = increased chance".format(correct_1,incorrect_1))
print("="*50)
