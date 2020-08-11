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
from scipy import spatial
from numba import jit

import drivinglab_projection as dp
from TrackMaker import TrackMaker

import sys
sys.setrecursionlimit(10000000)


    
#https://towardsdatascience.com/speed-up-your-algorithms-part-2-numba-293e554c5cc1

def closest_on_screen_point(trajectory, viewpoint, yaw, gaze_on_screen):


    """maps gaze on screen to the closest point on a given trajectory """

    traj_angles = dp.world_to_angles_through_screen(trajectory, viewpoint, yaw)        
    #pprint(traj_angles)

    #onscreen_idx, dists, *_ = find_closest_index(traj_angles, gaze_on_screen)
    #idx = closest_node(traj_angles, gaze_on_screen)
    idx = find_closest_index(traj_angles, gaze_on_screen)
   # print(idx)

    #traj_ref = trajectory[idx, :]
    screen_ref = traj_angles[idx, :]
    world_ref = trajectory[idx, :]

    path_dist = ab_path_length(trajectory, viewpoint, world_ref)
    path_dist /= 8.0 #time headway

    #plot_traj(screen_ref, gaze_on_screen, traj_angles)

    return(idx, screen_ref, world_ref, path_dist)#, traj_angles)


def plot_traj(screen_ref, gaze_on_screen, traj_angles):

    plt.cla()
    #print(screen_ref, gaze_on_screen)

    angles_limits_bottom = dp.screen_to_angles([0,0])[0]
    angles_limits_top = dp.screen_to_angles([1,1])[0]

    plt.xlim(angles_limits_bottom[0],angles_limits_top[0])
    plt.ylim(angles_limits_bottom[1],angles_limits_top[1])

    plt.plot(traj_angles[:,0],traj_angles[:,1], 'C3o', markersize = .5, )
    plt.plot(screen_ref[0],screen_ref[1], 'C1o', markersize = 5, markeredgecolor = 'k')
    plt.plot(gaze_on_screen[0],gaze_on_screen[1], 'mo', markersize = 5, markeredgecolor = 'k')    

    plt.pause(.1)
    
    #plt.show()

def closest_on_screen_point_optim(trajectory, viewpoint, yaw, gaze_on_screen):


    """maps gaze on screen to the closest point on a given trajectory """
    
    traj_angles = dp.world_to_angles_through_screen(trajectory, viewpoint, yaw)        
    
    #pprint(traj_angles)

    dist, idx = closest_node_tree(traj_angles, gaze_on_screen)
    ml_screen_ref = traj_angles[idx]    

    return(idx, ml_screen_ref)


def closest_node_vec(ref_array, nodes):
	#https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    #nodes = np.asarray(nodes)
	nodes = nodes[:, np.newaxis]
	print(nodes.shape)
	print(ref_array.shape)
	deltas = ref_array - nodes
	print(deltas.shape)
	#dist_2 = np.einsum('ij,ij->i', deltas, deltas)
	#print(dist_2.shape)
	#print(dist_2)
	#mins = np.argmin(dists, axis = 0)
	hehe
	return mins

def closest_node_tree(ref_array, nodes):

	mytree = spatial.cKDTree(ref_array)
	dist, idx = mytree.query(nodes)
	return dist, idx

def closest_node(ref_array, node):
	#https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    #nodes = np.asarray(nodes)
    deltas = ref_array - node
    #dist = np.einsum('ij,ij->i', deltas, deltas)
    dist = np.sqrt(deltas[:,0]**2 + deltas[:,1]**2)
    minidx = np.argmin(dist)

    return minidx#, dist[minidx]

@jit
def closest_node_numba(ref_array, node):
	#https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
    #nodes = np.asarray(nodes)
    deltas = ref_array - node
    dist = np.sqrt(deltas[:,0]**2 + deltas[:,1]**2)
    #   dist = np.sqrt(np.einsum('ij,ij->i', deltas, deltas))
    minidx = np.argmin(dist)

    return minidx, dist[minidx]
    
def find_closest_index(traj, point):
	"""
	returns index in traj of closest index to point

	"""

	#TODO: vectorise function to receive any length of points.

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

	return idx#, dists, dist
	#return idx
    
def ab_path_length(trajectory, a, b):
    """
    traj is the x,z future path.
    a,b are the 2D start nad end points
    """
    a_idx = find_closest_index(trajectory, a)
    b_idx = find_closest_index(trajectory, b)
    path = trajectory[a_idx:b_idx]
    step_dists = np.linalg.norm(np.diff(path, axis=0, prepend = np.array([[0,0]])), axis = 1)
    path_length = np.sum(step_dists)

    return(path_length)
        

def main(datafilepath):

    """
    1) for approach calc density of gaze point on midline, global point.
    2) for bends map target to midline, or gaze density (should be similar), global.
    3) calc TH to entry, and TH to target point.


    """
    #create midline
    sectionsize = 10000
    TrackData = TrackMaker(sectionsize) # 10000
    moving_window = sectionsize*2
    midline = TrackData[0]  
    sections = TrackData[2]
    #midline = midline[sections[0]:sections[5],:]  #only work with the midline of the trial     
    #steergaze_df = pd.read_feather(datafilepath)
    steergaze_df = pd.read_csv(datafilepath, sep=',',header=0)
    #steergaze_df.reset_index()
    master_steergaze = pd.DataFrame()
    datafolder = os.path.split(datafilepath)[0]  

    #TODO: due to grouping the future path cuts - off at end of slalom, use the continuous trajectory across roadsections for fp mapping

    #modes taken from gaze_through_midline_densities.py
    entry = find_closest_index(midline, [-23, 69])
    firstobject = find_closest_index(midline, [25, 52])
    gazemodes = [entry, firstobject]

    mid_diff = np.linalg.norm(np.diff(midline, axis=0, prepend = np.array([[0,0]])), axis = 1)
    midline_dist_array = np.cumsum(mid_diff)

    tree = spatial.cKDTree(midline)

    #for trial in picked_trials:	
    for block, blockdata in steergaze_df.groupby(['ID','block']):

        print(block)
        begin = timer()


        blockdata = blockdata.copy()
        blockdata.sort_values('currtime', inplace=True)
       # blockdata.reset_index()

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

        yaw = blockdata['yaw'].values
        
        #gaze_on_screen = blockdata['hangle'].values, blockdata['vangle'].values
        gaze_on_screen = np.transpose(np.array([blockdata['hangle'].values, blockdata['vangle'].values]))

        #print(yaw[0])
        #index = i
        #	viewpoint = blockdata['posx'].values, blockdata['posz'].values
        roadsection = blockdata['roadsection'].values

        #find time headway along MIDLINE  
        """
        start = timer()
        #idx, *_ = find_closest_index(midline, trajectory[0,:])
        idx = [find_closest_index(midline, viewpoint) for viewpoint in trajectory] 
        print(idx[:10])
        print(timer()-start)
        """

        #closest_indexes = [closest_node(midline, viewpoint) for viewpoint in trajectory] 
    #closest indexes
        #print(np.take(midline, 5, axis = 0, mode = 'wrap'))
        #print(np.take(midline, len(midline), axis = 0, mode = 'wrap'))
        #print(np.take(midline, 0, axis = 0, mode = 'wrap'))
        _, closest_indexes = tree.query(trajectory) 

        end_of_view = closest_indexes + moving_window

        #futuremid = np.take(midline, range(closest_indexes[0], end_of_view[0]), axis = 0, mode = 'wrap')
        def takemid(c,e):
            return (np.take(midline, range(c, e), axis = 0, mode = 'wrap'))

        start = timer()
        ml_idx, ml_screen_refs, ml_world_refs, ml_th = zip(*[
            closest_on_screen_point(takemid(c,e), t, y, g) 
            for c, e, t, y, g in zip(closest_indexes, end_of_view, trajectory, yaw, gaze_on_screen)
            ])
        print(timer() - start)        
        
        print(ml_screen_refs.shape)
        print(type(ml_screen_refs))
        ml_screen_refs = ml_screen_refs.reshape(-1, 2)
        ml_world_refs = ml_world_refs.reshape(-1, 2)
        print(ml_th)

        blockdata['midline_ref_onscreen_x'] = ml_screen_refs[:, 0]
        blockdata['midline_ref_onscreen_z'] = ml_screen_refs[:, 1]
        blockdata['midline_ref_world_x'] = ml_world_refs[:, 0]
        blockdata['midline_ref_world_z'] = ml_world_refs[:, 1]
        blockdata['th_along_midline'] = ml_th

        #find closest point on FUTURE PATH, with th calc along the path 
        
        traj_index = range(len(trajectory))
        fp_idx, fp_screen_refs, fp_world_refs, fp_th = zip(*[
            closest_on_screen_point(trajectory[i:(i+1000),:], t, y, g)    
            for i, t, y, g in zip(traj_index, trajectory, yaw, gaze_on_screen)
            ])
        #future_traj = trajectory[index:(index+window_fp), :]
        #fp_world_ref, fp_idx, dists, fp_angles = closest_on_screen_point(future_traj, viewpoint, yaw, gaze_on_screen)
        print(fp_screen_refs.shape)
        print(type(fp_screen_refs))
        fp_screen_refs = fp_screen_refs.reshape(-1, 2)
        fp_world_refs = fp_world_refs.reshape(-1, 2)
        print(ml_th)

        blockdata['futurepath_ref_onscreen_x'] = fp_screen_refs[:, 0]
        blockdata['futurepath_ref_onscreen_z'] = fp_screen_refs[:, 1]
        blockdata['futurepath_ref_world_x'] = fp_world_refs[:, 0]
        blockdata['futurepath_ref_world_z'] = fp_world_refs[:, 1]
        blockdata['th_along_futurepath'] = fp_th
        
        

        #TODO: current method runs into problems if the viewpoint is just before the midline resets (i.e. very large midline_dist_array value).
        #but not a problem for current analysis because trial starts from beginning of midline.
        #th_to_entry
        mid_dist_viewpoint = midline_dist_array[idx]

        mid_dist_entry = midline_dist_array[gazemodes[0]]
        th_to_entry = (mid_dist_entry - mid_dist_viewpoint) / 8.0 #if it's negative you have passed the point
        blockdata.loc[index,'veh_th_to_entry'] = th_to_entry

        #th_to_object
        mid_dist_object = midline_dist_array[gazemodes[1]]
        th_to_object = (mid_dist_object - mid_dist_viewpoint) / 8.0 #if it's negative you have passed the point
        blockdata.loc[index,'veh_th_to_object'] = th_to_object		
        
        """
        trialcode = row['trialcode']
        #plot			        
        #print("th_along_midline", ml_timeheadway)
        #print('ml_ref', ml_world_ref)
        #print("th_along_futurepath", fp_timeheadway)
        #print("fp_ref", fp_world_ref)

        world_gaze = dp.angles_to_world(gaze_on_screen, viewpoint, yaw)
        #print("world_gaze", world_gaze)

        plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
        plt.xlim(angles_limits_bottom[0],angles_limits_top[0])

        plt.plot(ml_angles[:,0],ml_angles[:,1], 'C3o', markersize = .5, )
        plt.plot(fp_angles[:,0],fp_angles[:,1], 'C2o', markersize = .5)
        plt.plot(ml_screen_ref[0],ml_screen_ref[1], 'C1o', markersize = 5, markeredgecolor = 'k')
        plt.plot(fp_screen_ref[0],fp_screen_ref[1], 'C0o', markersize = 5, markeredgecolor = 'k')

        plt.plot(gaze_on_screen[0],gaze_on_screen[1], 'mo', markersize = 5, markeredgecolor = 'k')
        plt.title(str(trialcode))


        plt.pause(.016)                    
        plt.cla()

        plt.show()
        """
		
        #master_steergaze = pd.concat([master_steergaze, blockdata])


        compute_time = timer()-begin
        print("Processing block took %f seconds" % compute_time)


        print("APPENDING  DATA FRAME")
        outfilepath = datafolder + '/trout_gazeandsteering_addthfrompath2.csv'

        with open(outfilepath, 'a', newline = '') as sgfile:
            blockdata.to_csv(sgfile, mode='a', header=sgfile.tell()==0)

        #master_steergaze.to_csv(datafolder + '/trout_gazeandsteering_addthfrompath.csv')

        #master_steergaze.to_feather(datafilepath)
	


if __name__ == '__main__':

    #datafilepath = "../Data/trout_rerun.feather"
	datafilepath = "../Data/trout_gazeandsteering_161019_addsample.csv"
	main(datafilepath)
