import pandas as pd
from TrackMaker import TrackMaker
from scipy import spatial
import sys
import numpy as np
import matplotlib.pyplot as plt
import feather


def main():
	sys.setrecursionlimit(10000000)

	sectionsize = 10000
	TrackData = TrackMaker(sectionsize) 
	midline = TrackData[0]  

	mid_diff = np.linalg.norm(np.diff(midline, axis=0, prepend = np.array([[0,0]])), axis = 1)
	midline_cumdist = np.cumsum(mid_diff)

	tree = spatial.cKDTree(midline)

	df = pd.read_csv('../Data/trout_twodatasets_full.csv')
	#df = df.query('roadsection < 2').copy() #not slalom.

	master_df = pd.DataFrame()

	drivingmodes = [0,1,2]
	for drive in drivingmodes:
		query_string = "drivingmode == {}".format(drive)     
		
		data = df.query(query_string).copy()

		for g, d in data.groupby(['ID']): 

			d = d.copy()
			d.sort_values('currtime', inplace=True)
			
			#find vehicle position on midline
			posx = d.posx.values
			posz = d.posz.values
			mirror = d.startingposition.values
			posx = posx * mirror
			posz = posz * mirror

			traj = np.transpose(np.array([posx, posz]))
			_, closest_idxs = tree.query(traj)
			
			#therefore find distances of mirrored position on cumdist array.
			cumdists = midline_cumdist[closest_idxs]
			x = cumdists
			
			#find th_along_midline
			y = d.th_along_midline.values
			
			#plt.plot(x,y, 'o', alpha = .1)        
			#plt.show()

			d.loc[:, 'midline_cumdist'] = cumdists

			print("appending block", g)
			master_df = pd.concat([master_df, d])

	master_df.to_csv 
#feather.write_dataframe(master_df, '../Data/trout_4.feather')    