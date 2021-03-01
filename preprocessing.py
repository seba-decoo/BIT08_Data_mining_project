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

#loop over attrutes and plotting 2 histogram on top of each other

for at in attr: 
    plt.figure()
    plt.hist(df0[at])
    plt.hist(df1[at])
    plt.show()
'''


#histogram

print(attr)
#plot histograms

# plot with various axes scales
plt.figure()

for i in attr:
    #plt.subplot()
    plt.hist(ds[i])
    #plt.grid(True)
    

plt.show()

'''


