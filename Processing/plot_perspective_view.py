import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
from file_methods import *
import cv2
from pprint import pprint

import drivinglab_projection as dp
from scipy.ndimage import gaussian_filter
from TrackMaker import TrackMaker

from scipy import spatial
import sys
sys.setrecursionlimit(10000000)

track = pd.read_csv("../Data/track_with_edges.csv")
inside_edge = track['insidex'].values, track['insidez'].values
outside_edge = track['outsidex'].values, track['outsidez'].values

angles_limits_bottom = dp.screen_to_angles([0,0])[0]

#print(angles_limits_bottom)
angles_limits_top = dp.screen_to_angles([1,1])[0]

pixels_limits_top = [1920,1080]

viewpoint = (-25, 40)
yaw = 0      
#compute track from viewpoint.
inside_edge_angles,depth = dp.world_to_angles_through_screen(np.transpose(inside_edge), viewpoint, yaw)
outside_edge_angles,depth = dp.world_to_angles_through_screen(np.transpose(outside_edge), viewpoint, yaw)    

#remove any above the horizon
inside_edge_angles = inside_edge_angles[depth > 0,:]
outside_edge_angles = outside_edge_angles[depth > 0,:]

#plot angles
#colors = ['xkcd:orange', 'xkcd:orange','xkcd:green', 'xkcd:green', 'xkcd:purple', 'xkcd:purple', 'xkcd:magenta', 'xkcd:magenta']
color_dict = {0: 'xkcd:turquoise', 35: 'xkcd:periwinkle'}
marker_dict = {-1.5: 'o', -.5: 'o', -10:'D',-9:'D'}

cmap = cm.get_cmap('tab10')
rgbas = cmap([0,2,1,3,4])
#colors = ['xkcd:orange', 'xkcd:orange','xkcd:green', 'xkcd:green']
angles = [[0,-1.5],
[0,-.5],
[0,-10],
[0,-9]]


def lookahead(x,y): 
		lookaheads = [np.sqrt(xi**2 + yi**2) / 8.0 for xi,yi in zip(x,y)]           
		return lookaheads

def addintersectinglines(ax, X, Y, ys = -2, col = 'b', al = .8):

		al = al * .8
		ymin, ymax = ax.get_ylim()
		yrange = ymax - ymin
		xmin, xmax = ax.get_xlim()
		xrange = xmax - xmin
		for y in list(ys):
			idx = np.argmin( abs(Y - y))
			x = X[idx]
			
			ax.axhline(y, xmin = 0, xmax = 1 - (xmax - x) / xrange, linestyle = ':', color = col, alpha = al)
			ax.axvline(x, ymin = 0, ymax = 1- (ymax - y) / yrange, linestyle = ':', color = col, alpha = al)

def addpoint(ax, X, Y, y, mec, mfc, al):
		
		idx = np.argmin( abs(Y - y))
		x = X[idx]        
		ax.plot(x, y, mec = mec,  mfc=mfc, marker = 'o', ms = 5, alpha = al, zorder = 3)

def plot_projection_contours(ax):

	   
	#ax.plot(inside_edge_angles[:,0], inside_edge_angles[:,1], 'k-', alpha = .6)
	#ax.plot(outside_edge_angles[:,0], outside_edge_angles[:,1], 'k-', alpha = .6)    
	#ax.ylim(angles_limits_bottom[1],angles_limits_top[1])
	#ax.xlim(angles_limits_bottom[0],angles_limits_top[0])

	h= np.linspace(-50, 50, 100)
	v = np.linspace(-30, -0, 100)
	

	H, V = np.meshgrid(h, v)


	coords = np.array([H.ravel(), V.ravel()]).T
	xpog, zpog =  dp.angles_to_world(angles = coords, position = viewpoint, yaw = yaw)
	TH = np.array(lookahead(xpog,zpog)).reshape(len(h),len(v))
	levels = np.arange(0,15, 1)
	ax.contour(H, V, TH, levels = levels, colors = "xkcd:light grey")
	#ax.show()	
	return (ax)
	#print(hist)
	
	"""
	hist = gaussian_filter(hist, 20)
	hist /= np.sum(hist)
	flat = np.sort(hist.ravel())[::-1]
	cum = np.cumsum(flat)
	level = [flat[cum > 0.8][0],flat[cum > 0.5][0]]
	print(hist.shape)
	print(len(h), len(v))
	print(level)
	H, V = np.meshgrid(h[:-1], v[:-1])
	"""
	#plt.contour(H, V, hist.T, levels=[level], colors=cmaps[mode])
	


def plot_projection_estimation():	
	
	fig_cm = np.array([13,9])
	fig_inc = fig_cm /2.54 
	fig, ax = plt.subplots(figsize=fig_inc, constrained_layout = True)    
	
	ax.plot(inside_edge_angles[:,0], inside_edge_angles[:,1], 'k-', alpha = .6)
	ax.plot(outside_edge_angles[:,0], outside_edge_angles[:,1], 'k-', alpha = .6)    

	
	for i, a in enumerate(angles):
		ax.plot(a[0], a[1], mec = rgbas[i], mfc = rgbas[i], marker = 'o', ms = 4)
		
		
	ax.set_ylim(angles_limits_bottom[1],angles_limits_top[1])
	ax.set_xlim(angles_limits_bottom[0],angles_limits_top[0])
	
	ax = plot_projection_contours(ax)
	#plt.axis("off")  

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	# Only show ticks on the left and bottom spines
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	for t in ax.get_xticklabels(): t.set_fontsize(8)
	for t in ax.get_yticklabels(): t.set_fontsize(8)
	ax.set_xlabel("Horizontal Angle ($^\circ$)", fontsize = 10)
	ax.set_ylabel("Vertical Angle ($^\circ$)", fontsize = 10)

	
	#add points
	left, bottom, width, height = [0.3, 0.7, 0.6, 0.25]
	ax2 = fig.add_axes([left, bottom, width, height])
	ax2.set_ylim(ymin = -11, ymax = 0)
	ax2.set_xlim(xmin = 0, xmax = 20)
	
	V = np.linspace(-15, -.3, 500)
	Hs = [np.zeros(len(V))]	

	for i, h in enumerate(Hs):
		coords = np.array([h, V]).T    
		xpog, zpog =  dp.angles_to_world(coords, position = viewpoint, yaw = yaw)    

		TH = np.array(lookahead(xpog,zpog))
		ax2.plot(TH, V, color = (.4,.4,.4), zorder = -1)		
		
	
	ths = []	
	for i, a in enumerate(angles):
		w = dp.angles_to_world(np.array(a))
		th = lookahead(w[0], w[1])

		print("angles: ", a)
		print("time headway: ", th)
		
		addintersectinglines(ax2, TH, V, [a[1]], rgbas[i], 1)
		ax2.plot(th, a[1], mec = rgbas[i],  mfc=rgbas[i], marker = 'o', ms = 4, zorder = 3)
			
		
		
		ths.append(th[0])
	
	yticks = [y for y in range(-10,0,2)]
	xticks = [x for x in range(5, 25, 5)]
	ax2.set_xticks(xticks)
	ax2.set_xticklabels(xticks, fontsize = 8)
	ax2.set_yticks(yticks)    
	ax2.set_yticklabels(yticks, fontsize = 8)
	ax2.set_ylabel(r"V$\theta$ ($^\circ$)", fontsize = 8)
	ax2.set_xlabel("Time Headway (s)", fontsize = 8)
	plt.savefig('projection_error_examples.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.savefig('projection_error_examples.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show()

def plot_track():

	fig_cm = np.array([13,8])
	fig_inc = fig_cm /2.54
	#fig,axes = plt.subplots(1, 3, constrained_layout = True, figsize = fig_inc)
	fig = plt.figure(constrained_layout = True, figsize = fig_inc)

	#last pos
	last_pos = np.array([9.50674491, 82.75534034]).T
	
	targets = pd.read_csv("../Data/TargetPositions.csv")
	target_centres = targets.loc[targets['condition']==0]
	target_circles = dp.target_position_circles(target_centres) #create position arrays	

	omit_color = (.8,.8,.8)

	plt.plot(track.insidez.values, track.insidex.values, '-', color = omit_color)
	plt.plot(track.outsidez.values, track.outsidex.values, 'k-', color = omit_color)

	

	for t in target_circles:
		t = np.squeeze(np.array(t)).T
		plt.plot(t[:,1], t[:,0], color = (.8,.8,.8), markersize = .2)

	#mean last pos
	inside = np.array([track.insidex.values, track.insidez.values]).T
	outside = np.array([track.outsidex.values, track.outsidez.values]).T

	#find indexes
	start = np.array([-25, 20]).T
	interp = np.array([-25, -20]).T
	for edge in [inside, outside]:
		for mirror in [1, -1]:						
			start_i, *_ = find_closest_index(edge, start * mirror)
			end_i, *_ = find_closest_index(edge, last_pos * mirror)
			arr = np.take(edge, range(start_i, end_i), axis = 0, mode = 'wrap')
			plt.plot(arr[:,1], arr[:,0], 'k')

			interp_i, *_ = find_closest_index(edge, interp * mirror)	
			if start_i < interp_i: start_i += edge.shape[0]	
			arr = np.take(edge, range(interp_i, start_i), axis = 0, mode = 'wrap')		
			plt.plot(arr[:,1], arr[:,0], color = omit_color)

	plt.plot(last_pos[1], last_pos[0], 'o', markersize = 3, color = 'r')
	plt.plot(-last_pos[1], -last_pos[0], 'o', markersize = 3, color = 'r')
	plt.xlabel('World Z (m)', fontsize = 10)
	plt.ylabel('World X (m)', fontsize = 10)

	plt.vlines(x = 20, ymin = -30, ymax = -20, color = 'k', linestyle = '-')
	plt.vlines(x = -20, ymin = -30, ymax = -20, color = omit_color, linestyle = '-')
	plt.vlines(x = 20, ymin = 30, ymax = 20, color = omit_color, linestyle = '-')
	plt.vlines(x = -20, ymin = 30, ymax = 20, color = 'k', linestyle = '-')

	plt.xlim(100,-100)	
	plt.axis('equal')
	plt.savefig('track.svg', format='svg', dpi=800, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show()


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

def plot_midline_estimation():
	#add midline points
	def closest_on_screen_point(traj_angles, gaze_on_screen):

		"""maps gaze on screen to the closest point on a given trajectory """		
		
		#pprint(traj_angles)

		onscreen_idx, dists, *_ = find_closest_index(traj_angles, gaze_on_screen)
		ref = traj_angles[onscreen_idx]

		return(ref)

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
		
	fig, ax = plt.subplots()
	TrackData = TrackMaker(10000)
	midline = TrackData[0]  
	tree = spatial.cKDTree(midline)
	_, idx = tree.query(viewpoint)
	future_mid = np.take(midline, range(idx, idx+20000), axis = 0, mode = 'wrap')

	marker_outline_viz = np.array([dp.bottom_left_pix,dp.bottom_right_pix,dp.top_right_pix,dp.top_left_pix,dp.bottom_left_pix],dtype=np.float32)
	marker_norm = np.divide(marker_outline_viz, dp.screen_res)
	screen_outline = dp.screen_to_angles(marker_norm)	
	traj_angles, depth = dp.world_to_angles_through_screen(future_mid, viewpoint, yaw)
	traj_angles = traj_angles[depth>0,:]

	def add_bounds(arr, bounds = 5):
		g = np.gradient(arr, axis = 0)
		angles = np.arctan2(g[:,1], g[:,0])
	
		#rotate point of on x,y graph using angles.	
		limits = []
		for rotate in [np.pi/2, np.pi]:
			angles += rotate #perpendicular normal. rotate counterclockwise
			#unit_normals = [np.cos(angles) - np.sin(angles), np.sin(angles) + np.cos(angles)] #on unit circle
			unit_normals = np.array([np.cos(angles), np.sin(angles)]) #on unit circle
			unit_normals *= bounds #in degrees?				
			unit_normals = unit_normals.T
			limits.append(arr + unit_normals)			

		return (limits)

	limits = add_bounds(traj_angles, 20)	
	ax.plot(traj_angles[:,0], traj_angles[:,1], 'k--', alpha = .8)	
	#limits[1] = limits[1][limits[1][:,0]>=20 & limits[1][:,1]>=-20,:]
	for i, l in enumerate(limits):

		if i == 1:
			l = l[(l[:,0] > 20) & (l[:,1] < -20), :]
		ax.plot(l[:,0], l[:,1], color = 'xkcd:grey', linestyle = '--', alpha = .8)	

	#ax.axhline(5, color = 'b', linestyle = '--')
	ax.plot(screen_outline[:,0], screen_outline[:,1], color = 'xkcd:grey', linestyle = '--')
	ax.plot(inside_edge_angles[:,0], inside_edge_angles[:,1], 'k-', alpha = .6)
	ax.plot(outside_edge_angles[:,0], outside_edge_angles[:,1], 'k-', alpha = .6)    
	ax.set_xlabel("Horizontal angle ($^\circ$)")
	ax.set_ylabel("Vertical angle ($^\circ$)")
	plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
	plt.xlim(angles_limits_bottom[0],angles_limits_top[0])


	#add axes for perc of data excluded
	left, bottom, width, height = [0.6, 0.7, 0.25, 0.15]
	ax2 = fig.add_axes([left, bottom, width, height])

	df = pd.read_feather('../Data/trout_subset_old.feather')
	query_string = "dataset == 2"
	df = df.query(query_string).copy()
	#% FALL WITHIN X DEGREES OF MIDLINE
	dists = np.sqrt(df.midline_vangle_dist.values ** 2 + df.midline_hangle_dist.values ** 2)
	total = len(dists)
	
	excl = np.linspace(0,20, num = 200)
	percs = []
	for crit in excl:
		nexcl = len(dists[dists>=crit])
		perc = nexcl/total
		percs.append(perc)
	
	ax2.plot(excl, percs)
	ax2.set_xlabel("Distance from midline reference ($^\circ$)", fontsize = 8)
	ax2.set_ylabel("Data excluded (%)", fontsize = 8)
	
	plt.savefig('mapping_method.svg', format='svg', dpi=800, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show()

	#TODO: ADD CUMULATIVE DISTANCE ALONG MIDLINE

#for annotated track
#plot_track()


#for appendix figure
plot_projection_estimation()


