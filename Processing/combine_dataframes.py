import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#df1 = pd.read_csv("../Data/trout18_gazeandsteering_2019-11-21.csv")
df1 = pd.read_csv("../Data/trout18_gazeandsteering_2019-12-06_1.csv")
print("loaded")
df2 = pd.read_csv("../Data/trout_rerun_gazeandsteering_2019-11-21.csv")
print("loaded")

df1['dataset'] = 1
df2['dataset'] = 2

master = pd.concat([df1, df2])

master.to_csv('../Data/trout_twodatasets.csv')
print("saved")
#master.to_feather('../Data/trout_twodatasets.feather')
#print("saved")

