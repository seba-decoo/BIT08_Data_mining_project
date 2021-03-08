import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import pandas

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

for att in dfnt:
    for a2 in dfnt:
        plt.scatter(dfnt[att],dfnt[a2],c=df["target"], cmap="prism")
        plt.xlabel(att)
        plt.ylabel(a2)
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



