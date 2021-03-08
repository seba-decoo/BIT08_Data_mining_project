import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import pandas
import matplotlib.gridspec as gridspec

# Reading in data from CSV
df = pandas.read_csv("heart.csv")

# Removing the class attribute for unsupervised clustering
dfnt = df.drop(columns=["target"])

#clustering
kmeans = cluster.KMeans(n_clusters=2).fit(dfnt)
y = kmeans.predict(dfnt)
dist = kmeans.transform(dfnt)

# script to go over all attributes and plot them against each other. try to do this for kmeans. can be compared to weka
print(dfnt)

a = 0
b = 0

plt.figure(figsize=[200,200])
for att in dfnt:
    for a2 in dfnt:
        plt.subplot2grid((14,14),(a,b))
        plt.scatter(dfnt[att],dfnt[a2],c=y, cmap="prism", s=5,marker=".")
        plt.xlabel(att, )# kan mss weg indien niet te vinden hoe size kan verbeterd worden
        plt.ylabel(a2) # zie boven
        plt.yticks([])
        plt.xticks([])
        #plt.show()
        b += 1
        if b == 13:
            b = 0
            a += 1

#plt.tight_layout()
plt.show()

"""
plt.scatter(dist[:,[0]],dist[:,[1]], c=y, cmap="PRGn")
plt.title("predicted target class")
plt.show()

plt.scatter(dist[:,[0]],dist[:,[1]], c=df["target"], cmap="PRGn")
plt.title("actual target class")
plt.show()

print(kmeans.cluster_centers_)
"""



