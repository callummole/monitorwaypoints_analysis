import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from file_methods import *
import cv2
from pprint import pprint

import drivinglab_projection as dp
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

def plot_gaze_density(TRACK, df, ax, cmaps):

	if TRACK == 1: 
		screenshot_path = "C:/git_repos/Trout18_Analysis/Processing/straight.bmp"
		df = df.query("currtimezero > 0 & currtimezero < 3")
	elif TRACK == 2: 
		screenshot_path = "C:/git_repos/Trout18_Analysis/Processing/bend.bmp"
		df = df.query("currtimezero > 8 & currtimezero < 12")
	else:
		raise Exception("invalid track")
	
	screenshot = cv2.imread(screenshot_path,1)
	screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB) 
	
	pixels_limits_bottom = [0,0]
	pixels_limits_top = [1920,1080]
	zord = {0: 3, 1: 2, 2: 1}
	for mode, md in df.groupby('drivingmode'):
		hlim, vlim = zip(pixels_limits_bottom, pixels_limits_top)
		da = 1
		h = np.arange(*hlim, da)
		v = np.arange(*vlim, da)
		#hist, _, _ = np.histogram2d(md.hangle.values, md.vangle.values, bins=(h, v))

		
		pixels_x = md.screen_x_pix.values 
		pixels_y = md.screen_y_pix.values 
		hist, _, _ = np.histogram2d(pixels_x, pixels_y, bins=(h, v))


		hist = gaussian_filter(hist, 15)
		
		
		hist /= np.sum(hist)
		flat = np.sort(hist.ravel())[::-1]
		cum = np.cumsum(flat)
		level = [flat[cum > 0.8][0]]#flat[cum > 0.5][0]]
		print(hist.shape)
		print(len(h), len(v))
		print(level)
		H, V = np.meshgrid(h[:-1], v[:-1])
		#plt.contour(H, V, hist.T, levels=[level], colors=cmaps[mode])
		ax.contour(H, V, hist.T, levels=level, colors=cmaps[mode], alpha = .8, zorder = zord[mode])
		#plt.plot(md['hangle'].values,md['vangle'].values, '.', alpha = .05) 
	"""
	if TRACK == 1: midway = df.query("currtimezero > 1.4 & currtimezero < 1.6")
	if TRACK == 2: midway = df.query("currtimezero > 9.9 & currtimezero < 10.1")

	posx_mean = np.median(midway.posx_mirror.values)
	print("posx_mean", posx_mean)
	posz_mean = np.median(midway.posz_mirror.values)
	print("posz_mean", posz_mean)
	yaw_mean = np.median(midway.yaw_mirror.values)
	print("yaw_mean", yaw_mean)
	viewpoint = posx_mean, posz_mean
	yaw = yaw_mean
	
	##load track
	track = pd.read_csv("../Data/track_with_edges.csv")
	inside_edge = track['insidex'].values, track['insidez'].values
	outside_edge = track['outsidex'].values, track['outsidez'].values
	
	#compute track from viewpoint.
	inside_edge_pixels, depth = dp.world_to_screen_homo_cave(np.transpose(inside_edge), viewpoint, yaw)
	outside_edge_pixels, depth = dp.world_to_screen_homo_cave(np.transpose(outside_edge), viewpoint, yaw)
	
	inside_edge_pixels = inside_edge_pixels[depth>0,:]
	outside_edge_pixels = outside_edge_pixels[depth>0,:]
			
	#plt.plot(inside_edge_pixels[:,0], inside_edge_pixels[:,1], 'k-')
	#plt.plot(outside_edge_pixels[:,0], outside_edge_pixels[:,1], 'k-')    

	"""

	

	ax.set_ylim(pixels_limits_bottom[1],pixels_limits_top[1])
	ax.set_xlim(pixels_limits_bottom[0],pixels_limits_top[0])

	#plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
	#plt.xlim(angles_limits_bottom[0],angles_limits_top[0])
	ax.imshow(cv2.flip(screenshot,0)) 
	ax.axis("off")  
	#ax.set_facecolor(background_color)

	"""
	if TRACK == 1:
		plt.savefig('densityplot_approach_1.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
		plt.savefig('densityplot_approach_1.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	if TRACK == 2:
		plt.savefig('densityplot_bend_1.png', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
		plt.savefig('densityplot_bend_1.png', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	"""
	
def plot_distributions(df, ax, cmaps):

	#th = df.th_along_midline.values
	x = np.linspace(0, 5, 500)

	zord = {0: 3, 1: 2, 2: 1}

	for g, d in df.groupby('drivingmode'):

		c = cmaps[g][0]
		gth = d.th_along_midline.values
		density = gaussian_kde(gth)		
		ax.plot(x,density(x), color = c, zorder = zord[g]) #row=0, col=0		

		
		mn = np.mean(gth)
		ax.axvline(ymin = 0, ymax = .2, x = mn, c = c, linestyle = '-')
		med = np.median(gth)
		ax.axvline(ymin = 0, ymax = .2, x = med, c = c, linestyle = ':')
		print("drivingmode: ", g)
		print("mean: ", mn)
		print("med: ", med)

	ax.set_xlabel('Time Headway (s)', fontsize = 12)    
	ax.set_ylabel('Density', fontsize = 12)    
	legend_elements = [Line2D([0], [0], color = cmaps[0][0], lw=4, label='Active'),
						Line2D([0], [0], color = cmaps[1][0], lw=4, label='Replay'),
						Line2D([0], [0], color = cmaps[2][0], lw=4, label='Stock')]
	ax.legend(handles=legend_elements, loc = [.7,.6])

if __name__ == '__main__':

	steergazefile = "../Data/trout_6.feather"
	steergaze = pd.read_feather(steergazefile)
	query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
	data = steergaze.query(query_string).copy()
	dists = np.sqrt(data.midline_vangle_dist.values ** 2 + data.midline_hangle_dist.values ** 2)

	data = data.iloc[dists<20,:]

#	data.sort_values('currtime', inplace=True)


	#https://www.sessions.edu/color-calculator/		
	#https://www.sessions.edu/color-calculator-results/?colors=03eeff,ffc403,ff0379
	active_color = tuple(np.array([3, 238, 255]) / 255)
	replay_color = tuple(np.array([255, 196, 3]) / 255)
	#replay_color = '#ff7f0e'
	stock_color = tuple(np.array([255, 3, 121]) / 255)
	cmaps = {0: [active_color], 1: [replay_color], 2: [stock_color]}

	ratio = 1080 / (1920*3)
	#fig_cm = np.array([18,18*ratio])
	fig_cm = np.array([13.2,6])
	fig_inc = fig_cm /2.54 
	
	#fig,axes = plt.subplots(1, 3, constrained_layout = True, figsize = fig_inc)
	fig = plt.figure(constrained_layout = True, figsize = fig_inc)
	gs = fig.add_gridspec(1,2)
	axes = [fig.add_subplot(gs[0, 0]),fig.add_subplot(gs[0, 1])]
	labels = ('A','B')
	for TRACK in [1,2]:		
		myax = axes[TRACK-1]
		plot_gaze_density(TRACK, data, myax, cmaps)
		myax.text(0.05, 0.95, labels[TRACK-1], transform=myax.transAxes,
      		fontsize=14, fontweight='bold', va='top')

	#ax = fig.add_subplot(gs[1,:])
	#ax.text(0.025, 0.95, labels[-1], transform=ax.transAxes,
    #  		fontsize=14, fontweight='bold', va='top')
	#plot_distributions(data, ax, cmaps)


	plt.savefig('overview.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.savefig('overview.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.savefig('overview.eps', format='eps', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')

	plt.show()
	
	
	
	
	

					