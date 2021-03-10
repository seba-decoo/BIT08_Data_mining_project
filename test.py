#Gridpec 

from typing import ChainMap
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas
import sklearn.cluster as cluster
'''
df = pandas.read_csv("heart.csv")
print(df)

fig = plt.figure()
fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

a =plt.scatter(x=df['age'], y=df['age'])
b =plt.scatter(x=df['target'], y=df['age'])
c =plt.scatter(x=df['age'], y=df['age'])
d =plt.scatter(x=df['age'], y=df['age'])

ax1.plot(df['age'], df['target'], 'ok', markersize=3 )
ax2.plot(df['target'], df['age'], 'ok', markersize)
#ax3.plot(c)
#ax4.plot(d)

plt.show()
'''
############################################""

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
plt.show()
        
        