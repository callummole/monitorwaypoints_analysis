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
from nslr_hmm import *
from timeit import default_timer as timer
from glob import glob
from pathlib import Path
from pprint import pprint
import feather

import drivinglab_projection as dp
from TrackMaker import TrackMaker

from scipy import spatial
sys.setrecursionlimit(10000000)

angles_limits_bottom = dp.screen_to_angles([0,0])[0]
angles_limits_top = dp.screen_to_angles([1,1])[0]
	

def closest_on_screen_point(trajectory, viewpoint, yaw, gaze_on_screen):


	"""maps gaze on screen to the closest point on a given trajectory """
	
	traj_angles, depth = dp.world_to_angles_through_screen(trajectory, viewpoint, yaw)        
	
	#pprint(traj_angles)
	traj_angles = traj_angles[depth>0,:]
	trajectory = trajectory[depth>0,:]

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
		

def main(steergaze_df):
	
	"""
	1) for approach calc density of gaze point on midline, global point.
	2) for bends map target to midline, or gaze density (should be similar), global.
	3) calc TH to entry, and TH to target point.


	"""
	#create midline
	TrackData = TrackMaker(10000)
	midline = TrackData[0]  
	sections = TrackData[2]	
	#midline = midline[sections[0]:sections[5],:]  #only work with the midline of the trial     
	#steergaze_df = pd.read_feather(datafilepath)
	#steergaze_df = pd.read_csv(datafilepath, sep=',',header=0)
	#steergaze_df.reset_index()
	master_steergaze = pd.DataFrame()
	#datafolder = os.path.split(datafilepath)[0]  

	#TODO: due to grouping the future path cuts - off at end of slalom, use the continuous trajectory across roadsections for fp mapping

	#modes taken from gaze_through_midline_densities.py
	#gazemodes = np.array([[-23,69],[25,52]])
	#entry, *_ = find_closest_index(midline, [-23, 69])
	#firstobject, *_ = find_closest_index(midline, [25, 52])
	#gazemodes = [entry, firstobject]

	mid_diff = np.linalg.norm(np.diff(midline, axis=0, prepend = np.array([[0,0]])), axis = 1)
	midline_dist_array = np.cumsum(mid_diff)

	tree = spatial.cKDTree(midline)

	#for trial in picked_trials:	
	for block, blockdata in steergaze_df.groupby(['ID','block']):
		
		#ppid, b = block[0], block[1]
		
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

		window_fp = 1000

		#the enumerate bracket gives you the integer, the iterrow func gives you the pandas index.
		for i, (rowindex, row) in enumerate(blockdata.iterrows()):

			index = i
			yaw = row['yaw']
			viewpoint = row['posx'], row['posz']
			gaze_on_screen = row['hangle'], row['vangle']
			roadsection = row['roadsection']
			

			#find time headway along MIDLINE      
			#idx, *_ = find_closest_index(midline, viewpoint)
			_, idx = tree.query(viewpoint)
			
			#window_end = idx+(window_fp*0)
			#future_mid = np.take(midline, range(idx, window_end), axis = 0, mode = 'wrap')
			#TODO: frustrum culling.

			ml_world_ref, ml_idx, dists, ml_angles = closest_on_screen_point(midline, viewpoint, yaw, gaze_on_screen)

			#print(ml_idx)
			ml_screen_ref = ml_angles[ml_idx,:]
			blockdata.loc[rowindex,'midline_ref_onscreen_x'] = ml_screen_ref[0]
			blockdata.loc[rowindex,'midline_ref_onscreen_z'] = ml_screen_ref[1]

			blockdata.loc[rowindex,'midline_ref_world_x'] = ml_world_ref[0]
			blockdata.loc[rowindex,'midline_ref_world_z'] = ml_world_ref[1]

			blockdata.loc[rowindex,'midline_hangle_dist'] = dists[0]
			blockdata.loc[rowindex,'midline_vangle_dist'] = dists[1]

			#ml_length = ab_path_length(midline, midline[idx,:], midline[ml_idx,:]) 
			_, new_idx = tree.query(ml_world_ref)
			#print("viewpoint", viewpoint)
			#print("world_ref", ml_world_ref)
			
			if new_idx < idx: new_idx += midline.shape[0]
			#print(idx, new_idx)
			path = np.take(midline, range(idx, new_idx), axis = 0, mode = 'wrap')
			diff_path = np.diff(path, axis=0)
			step_distances = np.linalg.norm(diff_path, axis = 1)
			ml_length = np.sum(step_distances)
			ml_timeheadway = ml_length / 8.0
			#print(ml_timeheadway)
			blockdata.loc[rowindex,'th_along_midline'] = ml_timeheadway

			#find closest point on FUTURE PATH, with th calc along the path
			"""
			future_traj = trajectory[index:(index+window_fp), :]
			fp_world_ref, fp_idx, dists, fp_angles = closest_on_screen_point(future_traj, viewpoint, yaw, gaze_on_screen)

			fp_screen_ref = fp_angles[fp_idx,:]
			blockdata.loc[rowindex,'futurepath_ref_world_x'] = fp_world_ref[0]
			blockdata.loc[rowindex,'futurepath_ref_world_z'] = fp_world_ref[1]

			blockdata.loc[rowindex,'futurepath_ref_onscreen_x'] = fp_screen_ref[0]
			blockdata.loc[rowindex,'futurepath_ref_onscreen_z'] = fp_screen_ref[1]

			blockdata.loc[rowindex,'futurepath_hangle_dist'] = dists[0]
			blockdata.loc[rowindex,'futurepath_vangle_dist'] = dists[1]

			fp_length = ab_path_length(trajectory, future_traj[0,:], future_traj[fp_idx,:]) 
			fp_timeheadway = fp_length / 8.0
			blockdata.loc[rowindex,'th_along_futurepath'] = fp_timeheadway
			"""


			#add timeheadway to focal point. currently crudely based on road section
			
			#TODO: current method runs into problems if the viewpoint is just before the midline resets (i.e. very large midline_dist_array value).
			#but not a problem for current analysis because trial starts from beginning of midline.
			
			"""
			mirror = row['startingposition']
			_, modepos = tree.query(gazemodes * mirror)

			mid_dist_viewpoint = midline_dist_array[idx]

			#th_to_entry
			mid_dist_entry = midline_dist_array[modepos[0]]
			th_to_entry = (mid_dist_entry - mid_dist_viewpoint) / 8.0 #if it's negative you have passed the point
			blockdata.loc[rowindex,'veh_th_to_entry'] = th_to_entry

			#th_to_object
			mid_dist_object = midline_dist_array[modepos[1]]
			th_to_object = (mid_dist_object - mid_dist_viewpoint) / 8.0 #if it's negative you have passed the point
			blockdata.loc[rowindex,'veh_th_to_object'] = th_to_object		
			"""

			
			#trialcode = row['trialcode']
			#plot			        
			#print("th_along_midline", ml_timeheadway)
			#print('ml_ref', ml_world_ref)
			#print("th_along_futurepath", fp_timeheadway)
			#print("fp_ref", fp_world_ref)

			#print("world_gaze", world_gaze)
			"""
			plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
			plt.xlim(angles_limits_bottom[0],angles_limits_top[0])

			plt.plot(ml_angles[:,0],ml_angles[:,1], 'C3o', markersize = .5, )
			plt.plot(ml_screen_ref[0],ml_screen_ref[1], 'C1o', markersize = 5, markeredgecolor = 'k')

			plt.plot(gaze_on_screen[0],gaze_on_screen[1], 'mo', markersize = 5, markeredgecolor = 'k')
			plt.title(str(ml_timeheadway))


			plt.pause(.016)                    
			plt.cla()

		plt.show()
		"""
		
		
		master_steergaze = pd.concat([master_steergaze, blockdata])


		compute_time = timer()-begin
		print("Processing block took %f seconds" % compute_time)


		print("APPENDING  DATA FRAME")
		#outfilepath = datafolder + '/trout_twodatasets_th.csv'

		#with open(outfilepath, 'a', newline = '') as sgfile:
	#		blockdata.to_csv(sgfile, mode='a', header=sgfile.tell()==0)
	return (master_steergaze)


if __name__ == '__main__':

	#datafilepath = "../Data/trout_rerun.feather"
	datafilepath = "../Data/trout_twodatasets.csv"
	main(datafilepath)
