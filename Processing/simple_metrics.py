import pandas as pd
import numpy as np

df = pd.read_csv("../Data/trout_twodatasets.csv")
df = df.query('confidence > .6')

for g, d in df.groupby(['drivingmode']):

    print(g)
    vangles = d.vangle.values
    med = np.median(vangles)
    print("median: ", med)
