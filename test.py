#Gridpec 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas

df = pandas.read_csv("heart.csv")
print(df)

fig = plt.figure()
fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

gs = GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[4, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

plt.scatter(x=df['age'], y=df['age'])
plt.scatter(x=df['target'], y=df['age'])
plt.scatter(x=df['age'], y=df['age'])
plt.scatter(x=df['age'], y=df['age'])

plt.show()