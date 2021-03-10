####################################################################################################
#PREPROCESSING
####################################################################################################

#import packages
import pandas
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries import frequencies

#specify dataset
ds = pandas.read_csv("heart.csv")

#change data-type variabels (continuous to nominal)

attchange = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "target"]

for change in attchange:
    ds[change] = ds[change].astype("category")

#print(ds.info())


#make dataframe based on target (0,1)
df0= ds[ds["target"] == 0]
df1= ds[ds["target"] == 1]

#list of attributes and remove target
attr = list(ds.columns)
attr.pop(-1)

#creating grid image of all attributes with 2 overlapping histograms of df0/df1
#specify figuresize
plt.figure(figsize=[14,10])

a=0
b=0

for at in attr:
    plt.subplot2grid((4,4), (a,b))
    #if ds.dtypes[at] == "int64":
    plt.hist(df0[at], color="#24FF00")
    plt.hist(df1[at], color="#FF1700", alpha=0.5)
    plt.title(at)
    #plt.grid(False)
    '''
    else:
        df0_freq = df0.value_counts(at)
        df1_freq = df1.value_counts(at)
      
        plt.bar(df0_freq[0],df0_freq[1], color="#24FF00")
        plt.bar(df1_freq[0],df1_freq[1], color="#FF1700", alpha=0.5)
    '''
    b += 1
    if b == 4:
        b=0
        a += 1

plt.tight_layout() 
#plt.show() #show the figure
plt.savefig('Images/hist.png') #save figure in directory 


