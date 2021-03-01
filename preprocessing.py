####################################################################################################
#PREPROCESSING
####################################################################################################

#import packages
import pandas
import matplotlib.pyplot as plt
import numpy as np

#specify dataset
ds = pandas.read_csv("heart.csv")
#make dataframe based on target (0,1)
df0= ds[ds["target"] == 0]
df1= ds[ds["target"] == 1]

#list of attributes and remove target
attr = list(ds.columns)
attr.pop(-1)
'''
#loop over attributes and plotting 2 histogram on top of each other
for at in attr:
    plt.figure()
    plt.hist(df0[at])
    plt.hist(df1[at])
    plt.show()
    a = a+1
'''
#creating grid image of all attributes with 2 overlapping histograms of df0/df1
#specify figuresize
plt.figure(figsize=[14,10])

a=0
b=0

for at in attr:
    plt.subplot2grid((4,4), (a,b))
    plt.hist(df0[at])
    plt.hist(df1[at], alpha=0.5)
    plt.title(at)
    #plt.grid(False)
    b += 1
    if b == 4:
        b=0
        a += 1

plt.tight_layout() 
#plt.show() #show the figure
plt.savefig('Images/hist.png') #save figure in directory #save figure to Image directory


