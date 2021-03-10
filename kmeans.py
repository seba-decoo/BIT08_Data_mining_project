####################################################################################################
####################################################################################################
#KMEANS CLUSTERING (discribtive data mining)
####################################################################################################
####################################################################################################

##Import modules
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
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
'''
#elbow method
# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = cluster.KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(dfnt)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
#kneeLocator
kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)
print(kl.elbow)
'''
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
plt.figure(figsize=[10,10])
plt.style.use("fivethirtyeight")
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
while x < 2:
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
dist = kmeans.transform(dfnt)

##Script to go over all attributes and plot them against each other. try to do this for kmeans. can be compared to weka
#print(dfnt)

spec2 = GridSpec(ncols=13, nrows=13, wspace=0, hspace=0)
fig = plt.figure(figsize=[16,9])
a=0
for att in dfnt:
    for a2 in dfnt:
        ax = fig.add_subplot(spec2[a])
        plt.scatter(dfnt[att],dfnt[a2],c=y, cmap="prism", s=5,marker=".")
        #plt.xlabel(att, )# kan mss weg indien niet te vinden hoe size kan verbeterd worden
        #plt.ylabel(a2) # zie boven
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