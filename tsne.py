import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from numpy import random

df = pandas.read_csv("heart.csv")
y = df["target"]

a = 1
while a <= 5: #Don't kill pc!!!
    p = random.randint(1,high=1000)
    print("The perplexity for plot {} is {}".format(a,p))
    fig = plt.figure(a, figsize=(8,6))
    x_reduced = TSNE(n_components=2, perplexity=p,random_state=0).fit_transform(df)
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1],c=y ,cmap='brg')
    a += 1

plt.show()

