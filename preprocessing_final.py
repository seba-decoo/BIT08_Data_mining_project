####################################################################################################
#PREPROCESSING
####################################################################################################
####################################################################################################
##Import modules
import pandas
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries import frequencies

##Load dataset
ds = pandas.read_csv("heart.csv")

##Dataframe based on target (0,1)
df0= ds[ds["target"] == 0]
df1= ds[ds["target"] == 1]

##List of attributes + remove target
attr = list(ds.columns)
attr.pop(-1)
####################################################################################################
#Grid view of all attributes with 2 overlapping histograms of df0/df1
####################################################################################################

##Counter
a=0
b=0
##Figure size
plt.figure(figsize=[14,10])

for at in attr:
    plt.subplot2grid((4,4), (a,b))
    plt.hist(df0[at], color="#24FF00")
    plt.hist(df1[at], color="#FF1700", alpha=0.5)
    plt.title(at)
    #Determination of coordinates
    b += 1
    if b == 4:
        b=0
        a += 1

plt.tight_layout() 
#plt.show() #show the figure during execution
plt.savefig('Images/hist.png') #save figure in directory