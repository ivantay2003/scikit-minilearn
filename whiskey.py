#source : https://www.geeksforgeeks.org/project-scikit-learn-whisky-clustering/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
# >>>whisky.head(), iloc method to index a data frame by location.
# >>>whisky.iloc[0:10], we specified the rows from 0 - 9
# >>>whisky.iloc[0:10, 0:5], we specified the rows from 0 - 9 & columns from 0-5
# >>>whisky.columns
flavors = whisky.iloc[:, 2:14]

corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)
plt.figure(figsize=(10, 10))
plt.pcolor(corr_flavors)
plt.colorbar()
# >>>plt.savefig("corlate-whisky1.pdf")

corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10, 10))
plt.pcolor(corr_whisky)
plt.axis("tight")
plt.colorbar()
# >>>plt.savefig("corlate-whisky2.pdf")

plt.show()