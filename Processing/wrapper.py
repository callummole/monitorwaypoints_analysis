
from timeit import default_timer as timer
import numpy as np
import pandas as pd

"""
start = timer()

wait = timer() - start
hour = 3600
waittime = hour * 7

print(start)
print("waiting...")
while wait < waittime:
    wait = timer()-start

print('waited')
print(timer())
"""

#TODO: Streamline script to avoid all the saving.

def combine_dataframes()#df1 = pd.read_csv("../Data/trout18_gazeandsteering_2019-11-21.csv")
    


master.to_csv('../Data/trout_twodatasets.csv')
print("saved")
#master.to_feather('../Data/trout_twodatasets.feather')
#print("saved")

datafilepath = "../Data/trout_twodatasets_th.csv"
#datafilepath = "../Data/trout_gazeandsteering_161019_addsample.csv"	
outfile = 'trout_twodatasets_full.csv'
gazemodes = np.array([[-22.74,70.4],[25,50.39]])
import th_to_focal_point




print("running th_to_focal_point")
df = th_to_focal_point.main(datafilepath, gazemodes, outfile)
print("finished th_to_focal_point")

print("running save_midline_cumdist")
import save_midline_cumdist
df = save_midline_cumdist.main(df)
print("finished save_midline_cumdist, now saving df")
df.to_csv(outfile)    

datafilepath = '../Data/trout_twodatasets_full.csv'
print("running save to feather")
import save_to_feather
save_to_feather.main(datafilepath, 'trout_5')
print("finished save to feather")

datafilepath = '../Data/trout_5.feather'
print("running save to feather")
import save_subset
save_subset.main(datafilepath, 'trout_subset_2')    
print("finished save to feather")

print("running linmix")
import linmix
linmix.main(plot=False, outfile = 'linmix_res.csv', datafile = '../Data/trout_subset_2.feather')
print("finished linmix")

