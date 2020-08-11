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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import drivinglab_projection as dp
from scipy.stats import gaussian_kde

from TrackMaker import TrackMaker
from scipy.special import softmax
from scipy import spatial
from memoize import memoize

screen_res = 1920, 1080
sys.setrecursionlimit(10000000)

"""
plot the midline references for gaze to find salient anchors for the LAF component of the mixture model.
    
"""
    
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

	return idx


def ab_path_length(trajectory, b):
	"""
	traj is the x,z future path.
	a,b are the 2D start nad end points
	"""

	b_idx = find_closest_index(trajectory, b)
	path = trajectory[0:b_idx]
	diff_path = np.diff(path, axis=0)
	step_distances = np.linalg.norm(diff_path, axis = 1)
	path_length = np.sum(step_distances)

	return(path_length)


def find_modes(df):

	"""find modes for road sections, could be a continuous function of time"""

	def get_density_max(vals):

		print(vals.shape)
		density = gaussian_kde(vals)
		#density.covariance_factor = lambda : .025
		#density._compute_covariance()
		print("here")
		x_min, x_max = -30, 30
		y_min, y_max = 0, 100
		#steps = (grid_max - grid_min) / .1
		#grid = np.linspace(grid_min, grid_max, num = steps)
		X,Y = np.mgrid[x_min:x_max:500j, y_min:y_max:500j] #.1 steps
		positions = np.vstack([X.ravel(), Y.ravel()])
		print(positions)
		print("here2")
		Z = density(positions)
		pprint(Z)
		print(Z.shape)
		print(X.shape)
		x_shaped = np.reshape(X, Z.shape)
		
		max_Z = np.argmax(Z)
		max_X = np.reshape(X, Z.shape)[max_Z]
		max_Y = np.reshape(Y, Z.shape)[max_Z]
		print(max_Z)
		print(max_X, max_Y)
		Z = np.reshape(Z.T, X.shape)
		#max_val = x[np.argmax(y)]

		plt.contour(X,Y,Z)
		plt.plot(max_X, max_Y, 'ro')
		#plt
		print("here3")
		#plt.plot(max_val,y[np.argmax(y)], 'ro')
		plt.show()

		
		return ( [max_Y, max_Y])


	modes = []
	for roadsection in range(2):
		query_string = "roadsection == {}".format(roadsection)     
		data = df.query(query_string).copy()
	
		pogx = data.midline_ref_onscreen_x.values		
		pogz = data.midline_ref_onscreen_z.values
		midline_gazes = np.transpose(np.array([pogx, pogz]))
		midline_gazes = midline_gazes[~np.isnan(midline_gazes).any(axis=1)]
		
		mode = get_density_max(midline_gazes)

		modes.append(mode)

	return modes


def load_targets(datafolder):

	targets = pd.read_csv(datafolder + "/TargetPositions.csv")
	targets = targets.loc[targets['targetindex']>2,:]
	targets_mean = targets.groupby(['targetindex']).mean()
			#pprint(targets_mean)

	return targets_mean



def plot_wraparound(midline, y_dens):

	g = np.gradient(midline, axis = 0)

	angles = np.arctan2(g[:,1], g[:,0])
	
	#rotate point of on x,y graph using angles.
	#xs = norms
	#ys = np.zeros(len(norms))
	angles += np.pi/2 #perpendicular normal. rotate counterclockwise
	#unit_normals = [np.cos(angles) - np.sin(angles), np.sin(angles) + np.cos(angles)] #on unit circle
	unit_normals = [np.cos(angles), np.sin(angles)] #on unit circle
	margin = 2 
	dens_scale = 500
	plot_dens = y_dens * dens_scale
	unit_normals *= (margin + plot_dens)
	unit_normals = unit_normals.T
	wrap = midline + unit_normals

	plt.plot(midline[:,0],midline[:,1], 'k-')
	plt.plot(wrap[:,0],wrap[:,1], 'C0-')
	plt.plot(unit_normals[:,0],unit_normals[:,1], 'C0-')
	plt.axis('equal')
	plt.show()

def add_landmarks(targets_mean, midline_dist_array, tree):
	
	cols = {0: 'g', 1: 'b', 2: 'm'}
	for i, (_, row) in enumerate(targets_mean.iterrows()):        
		point = row['xcentre'], row['zcentre']
		_, idx = tree.query(point)
		dist = midline_dist_array[idx]
		print(str(i), dist)
		plt.axvline(x=dist, ls = '--', color = cols[i])

	_, idx = tree.query([-25, 60])
	straight_end = midline_dist_array[idx]
	print("st", straight_end)
	plt.axvline(x=straight_end, ls = '--', c = 'r')

	_, idx = tree.query([25, 60])
	bend_end = midline_dist_array[idx]
	print("be", bend_end)
	plt.axvline(x=bend_end, ls = '--', c = 'r')

def add_section_max(y_dens, x, plot = True, cutoff = 100):
	
	straight_mask = x<cutoff
	straight_dists = x[straight_mask]
	straight_dens = y_dens[straight_mask]
	bend_mask = x>cutoff
	bend_dists = x[bend_mask]
	bend_dens = y_dens[bend_mask]

	max_straight = straight_dists[np.argmax(y_dens[straight_mask])]
	max_bend = bend_dists[np.argmax(y_dens[bend_mask])]
	
	if plot:
		plt.plot(max_straight,straight_dens[np.argmax(y_dens[straight_mask])], 'ro')
		plt.plot(max_bend,bend_dens[np.argmax(y_dens[bend_mask])], 'bo')

	return (max_straight, max_bend)

def plot_heatmap_track(midline, y_dens, datafolder):

	
	#multi-coloured line
	# Create a set of line segments so that we can color them individually
	# This creates the points as a N x 1 x 2 array so that we can stack points
	# together easily to get the segments. The segments array for line collection
	# needs to be (numlines) x (points per line) x 2 (for x and y)
			
	#print("mid shape", midline.shape)
	midline[:] = midline[:,[1,0]]

	points = midline.reshape(-1, 1, 2)
	
	#print("points shape", points.shape)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	#print("segments shape", segments.shape)

	track = pd.read_csv(datafolder + "/track_with_edges.csv")

	fig_cm = np.array([12,7])
	fig_inc = fig_cm /2.54 
	fig = plt.figure(figsize=fig_inc, constrained_layout = True)

	# Create a continuous norm to map from data points to colors
	norm = plt.Normalize(y_dens.min(), y_dens.max())
	lc = LineCollection(segments, cmap='gray_r', norm=norm)
	# Set the values used for colormapping
	lc.set_array(y_dens)
	lc.set_linewidth(12)
	axs = plt.gca()
	line = axs.add_collection(lc)
	cbar = fig.colorbar(line, ax=axs, label = 'density', aspect = 40)
	cbar.ax.tick_params(labelsize=8)
	
	plt.plot(track.insidez.values, track.insidex.values, 'k-')
	plt.plot(track.outsidez.values, track.outsidex.values, 'k-')

	#buffer = 5
	#axs.set_xlim(midline[:,0].min() - buffer, midline[:,0].max() + buffer)
	#axs.set_ylim(midline[:,1].min() - buffer, midline[:,1].max() + buffer)

	yrng = 60
	xmin, ymin = 19, -30
	axs.set_xlim(xmin)
	axs.set_ylim(ymin,ymin+yrng)
	axs.set_xlabel("World Z (m)", fontsize = 10)
	axs.set_ylabel("World X (m)", fontsize = 10)
	for t in axs.get_xticklabels(): t.set_fontsize(8)
	for t in axs.get_yticklabels(): t.set_fontsize(8)

	plt.arrow(32, -25, 5, 0, head_width = 1, zorder = 4, fc = 'k') 
	axs.invert_yaxis()
	

	#plt.savefig('heatmap_track.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	#plt.savefig('heatmap_track.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	#plt.show()

@memoize
def estimate_density(x,y):

	density = gaussian_kde(y)
	#print("cov_fact", density.covariance_factor())
	density.covariance_factor = lambda : .025
	density._compute_covariance()	
	#print("calculating density...")
	y_dens = density(x)
	return(y_dens)

def main(datafilepath, plot = True):

	"""
	1) for approach calc density of gaze point on midline, global point.
	2) for bends map target to midline, or gaze density (should be similar), global.
	3) calc TH to entry, and TH to target point.
	"""

	#create midline
	datafolder = os.path.split(datafilepath)[0]  

	TrackData = TrackMaker(10000)
	midline = TrackData[0]  
	sections = TrackData[2]
	#midline = midline[sections[0]:sections[5],:]  #only work with the midline of the trial     

	modes = pd.DataFrame(columns = ['roadsection','drivingmode','mode_x','mode_z'])

	tree = spatial.cKDTree(midline)

	#for roadsection in range(2):
#		for drivingmode in [1]: #range(2):
	
	steergaze_df = pd.read_feather(datafilepath)
	#query_string = "drivingmode == {} & roadsection == {} & confidence > .8".format(drivingmode, roadsection)     
	#query_string = "drivingmode == 0"
	query_string = "drivingmode == 0 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2 "

	print(query_string)
	data = steergaze_df.query(query_string).copy()
	data.sort_values('currtime', inplace=True)

	#modes = find_modes(data)
	#print(modes)
	pogx = data.midline_ref_world_x.values
	pogz = data.midline_ref_world_z.values
	mirror = data.startingposition.values
	pogx = pogx * mirror
	pogz = pogz * mirror

	"""
	#build lines
	lines = [[(ml_x, ml_z), (g_x, g_z)] for ml_x, g_x, ml_z, g_z  in zip(mirrored_pogx, mirrored_xpog, mirrored_pogz, mirrored_zpog)]
	lc = LineCollection(lines, linewidths=.5,
							colors='k', linestyle='solid', alpha = .1)

	plt.plot(midline[:,0],midline[:,1], 'k-')
	#	plt.plot(mirrored_pogx, mirrored_pogz, 'g.', alpha = .1)
	plt.plot(pogx, pogz, 'b.', alpha = .1)

	ax = plt.gca()
	ax.add_collection(lc)

	plt.ylim(min(midline[:,1]) -5,max(midline[:,1]) +5)
	plt.xlim(min(midline[:,0]) -5,max(midline[:,0]) +5)
	plt.show()			

	"""

	midline_gazes = np.transpose(np.array([pogx, pogz]))
	midline_gazes = midline_gazes[~np.isnan(midline_gazes).any(axis=1)]

	mid_diff = np.linalg.norm(np.diff(midline, axis=0, prepend = np.array([[0,0]])), axis = 1)
	midline_dist_array = np.cumsum(mid_diff)

	_, closest_indexes = tree.query(midline_gazes) 
	mid_dists = midline_dist_array[closest_indexes]

	"""
	for i, mid_gaze in enumerate(midline_gazes):
		#print(mid_gaze)
		#mid_dists[i] = ab_path_length(midline, mid_gaze)
		pt_idx = find_closest_index(midline, mid_gaze)
		mid_dists[i] = midline_dist_array[pt_idx]
	"""
	#hacky way of finding anchors
	x = midline_dist_array
	y_dens = estimate_density(x, mid_dists)

	targets_mean = load_targets(datafolder)

	#plot density
	if plot:
		print("plotting")
		#plt.plot(x,y_dens)
		#add_landmarks(targets_mean, midline_dist_array, tree)
		#max_straight, max_bend = add_section_max(y_dens, x, plot = True)
		#plt.show()
		#plot_wraparound(midline, y_dens)
		plot_heatmap_track(midline, y_dens, datafolder)		

		#plot final position
		last_pos = np.array([9.50674491, 82.75534034]).T
		start = np.array([-25, 20]).T
		plt.plot(last_pos[1], last_pos[0], 'o', markersize = 3, color = 'r')
		plt.plot(start[1], start[0], 'o', markersize = 3, color = 'r')		

		plt.savefig('track_heatmap.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
		plt.savefig('track_heatmap.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
		plt.savefig('track_heatmap.eps', format='eps', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
		plt.show()
	else:
		max_straight, max_bend = add_section_max(y_dens, x, plot = False)

#	mode_straight = midline[midline_dist_array==max_straight,:]#
	#print("mode straight: ", mode_straight)
	#mode_bend = midline[midline_dist_array==max_bend,:]
	#print("mode bend:", mode_bend)

	#return ([mode_straight, mode_bend])

	

if __name__ == '__main__':

    datafilepath = "../Data/trout_6.feather"
    main(datafilepath, plot = True)
