###script to view and almagamate eye tracking files.
import numpy as np
import sys, os
from file_methods import *
import pickle
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import math as mt 
from scipy.interpolate import interp1d
from timeit import default_timer as timer
from glob import glob
from pathlib import Path
from pprint import pprint
import feather
import warnings

import drivinglab_projection as dp
from TrackMaker import TrackMaker

from scipy import spatial
sys.setrecursionlimit(10000000)

angles_limits_bottom = dp.screen_to_angles([0,0])[0]
angles_limits_top = dp.screen_to_angles([1,1])[0]

def check_exist(filepath):

	exists = True
	i = 1
	path, ext = os.path.splitext(filepath)		
	while exists:
		if os.path.exists(filepath):
			print("file exists, renaming...", filepath)
			filepath = path + '_' + str(i) + ext
			i += 1
		else:
			exists = False
	
	return(filepath)
		

def closest_on_screen_point(trajectory, viewpoint, yaw, gaze_on_screen):


	"""maps gaze on screen to the closest point on a given trajectory """
	
	traj_angles = dp.world_to_angles_through_screen(trajectory, viewpoint, yaw)        
	
	#pprint(traj_angles)

	onscreen_idx, dists, *_ = find_closest_index(traj_angles, gaze_on_screen)
	traj_ref = trajectory[onscreen_idx, :]

	return(traj_ref, onscreen_idx, dists, traj_angles)
	   
	
def find_closest_index(traj, point):
	"""
	returns index in traj of closest index to point

	"""
	distances =  np.subtract(np.array(point),traj) 
	distances = distances.reshape(-1,2)
	#distances = distances[~np.isnan(distances)].reshape(-1,2)

	#print("distances")
	#pprint(distances)
	dist_array = np.linalg.norm(distances, axis = 1)
	#pprint(dist_array)
	#dist_array = np.sqrt((distances[:,0]**2)+(distances[:,1]**2)) #array of distances from trajectory to gaze landing point in world.        
	idx = np.nanargmin(abs(dist_array)) #find smallest difference in pythag distance from 0,0 to get closest point.        
	dists = distances[idx, :]
	dist = dist_array[idx]

	return idx, dists, dist
	
def ab_path_length(trajectory, a, b):
	"""
	traj is the x,z future path.
	a,b are the 2D start nad end points
	"""

	a_idx, *_ = find_closest_index(trajectory, a)
	b_idx, *_ = find_closest_index(trajectory, b)
	path = trajectory[a_idx:b_idx]
	diff_path = np.diff(path, axis=0)
	step_distances = np.linalg.norm(diff_path, axis = 1)
	path_length = np.sum(step_distances)

	return(path_length)
		

def main(steergaze_df, gazemodes = [], outfile = None):
	"""
	1) for approach calc density of gaze point on midline, global point.
	2) for bends map target to midline, or gaze density (should be similar), global.
	3) calc TH to entry, and TH to target point.
	"""

	screen_res = 1920, 1080
	if len(gazemodes) == 0: raise Exception("must pass gaze modes")
	"""
	if outfile == None: raise Exception("no outfile given")
	
	#ensure don't overwrite
	#datafolder = os.path.split(datafilepath)[0]  
	outfilepath ='../Data/' + outfile
	outfilepath = check_exist(outfilepath)
	print(outfilepath)
	"""

	#create midline
	TrackData = TrackMaker(10000)
	midline = TrackData[0]  
	sections = TrackData[2]
	#midline = midline[sections[0]:sections[5],:]  #only work with the midline of the trial     
	#steergaze_df = pd.read_feather(datafilepath)
	"""
	_, ext = os.path.splitext(datafilepath)
	if 'feather' in ext:
		steergaze_df = pd.read_feather(datafilepath)
	elif 'csv' in ext:
		steergaze_df = pd.read_csv(datafilepath, sep=',',header=0)
	else:
		raise Exception("invalid datafilepath type")
	"""

	#steergaze_df.reset_index()
	master_steergaze = pd.DataFrame()
	

	mid_diff = np.linalg.norm(np.diff(midline, axis=0, prepend = np.array([[0,0]])), axis = 1)
	midline_dist_array = np.cumsum(mid_diff)

	tree = spatial.cKDTree(midline)

	#for trial in picked_trials:	
	print("beginning loop")
	for block, blockdata in steergaze_df.groupby(['ID','block']):
				
		print(block)
		begin = timer()


		blockdata = blockdata.copy()
		blockdata.sort_values('currtime', inplace=True)
		blockdata.reset_index(drop = True)

		####pick target
		"""
		condition = blockdata.condition.values[0]
		target_centres = targets.loc[targets['condition']==int(condition),:]
		#pprint(target_centres)

		target_centres = target_centres.reset_index(drop=True)
		#pick starting position.
		start_x = np.sign(blockdata['posx']).values[0]
		#select targets with opposite sign for xcentre, these will be the ones encountered in that block
		target_centres = target_centres.loc[np.sign(target_centres['xcentre'])!=start_x,:]                
		target_circles = dp.target_position_circles(target_centres)

		"""
		
		traj_x = blockdata['posx'].values
		traj_z = blockdata['posz'].values

		trajectory = np.transpose(np.array([traj_x, traj_z]))

		#the enumerate bracket gives you the integer, the iterrow func gives you the pandas index.
		for i, (rowindex, row) in enumerate(blockdata.iterrows()):

			index = i
			yaw = row['yaw']
			viewpoint = row['posx'], row['posz']
			roadsection = row['roadsection']			            
	
			_, idx = tree.query(viewpoint)
			mid_dist_viewpoint = midline_dist_array[idx]
									

			mirror = row['startingposition']

			if np.isnan(mirror): continue

			_, modepos = tree.query(gazemodes * mirror)


			cols_th = ['veh_th_to_entry', 'veh_th_to_object']
			cols_onscr = ['entry_on_screen','object_on_screen']
			
			#lafsections: 0 = entry on screen, 1 = neither on screen, 2 = object on screen
			#lafsections = {0: 0, 1: 2} #for ease of looping
			on_screen = [False, False]
			for gm_i, gm in enumerate(modepos):
				#mid_gm = midline[gm]

				#target_pixels = dp.world_to_screen_homo_cave(mid_gm, viewpoint, yaw)[0]
				#print(target_pixels)
				#bool_mask = (target_pixels[0] > 0) & (target_pixels[0] < screen_res[0]) & (target_pixels[1] > 0) & (target_pixels[1] < screen_res[1])
				#print(bool_mask)
				
				#on_screen[gm_i] = bool_mask
				#blockdata.loc[rowindex, cols_onscr[gm_i]] = on_screen[gm_i]
				#blockdata.loc[rowindex, 'lafsection'] = lafsections[gm_i]

				mid_dist_gm = midline_dist_array[gm]
				th_to_gm = (mid_dist_gm - mid_dist_viewpoint) / 8.0 #if it's negative you have passed the point
				blockdata.loc[rowindex,cols_th[gm_i]] = th_to_gm

				

			if not any(on_screen): blockdata.loc[rowindex, 'lafsection'] = 1
			if all(on_screen): warnings.warn("both gaze modes on screen, something has went wrong", block, index)
				
		master_steergaze = pd.concat([master_steergaze, blockdata])

		compute_time = timer()-begin
		print("Processing block took %f seconds" % compute_time)

		#print("APPENDING  DATA FRAME")		
		#with open(outfilepath, 'a', newline = '') as sgfile:
	#		blockdata.to_csv(sgfile, mode='a', header=sgfile.tell()==0)
	return(master_steergaze)

if __name__ == '__main__':

	datafilepath = "../Data/trout_twodatasets_th.csv"
	#datafilepath = "../Data/trout_gazeandsteering_161019_addsample.csv"	
	outfile = 'trout_twodatasets_full.csv'
	gazemodes = np.array([[-22.74,70.4],[25,50.39]])
	main(datafilepath, gazemodes, outfile)
