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

plt.scatter(dist[:,[0]],dist[:,[1]], c=y, cmap="PRGn")
plt.show()

plt.scatter(dist[:,[0]],dist[:,[1]], c=df["target"], cmap="PRGn")
plt.show()

print(kmeans.cluster_centers_)






