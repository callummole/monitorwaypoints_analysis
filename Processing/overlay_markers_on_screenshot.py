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

def plot_screenshot(ax):

	
	screenshot_path = "C:/git_repos/Trout18_Analysis/Processing/straight.bmp"
	screenshot = cv2.imread(screenshot_path,1)
	screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)


	ax.imshow(cv2.flip(screenshot,0)) 	

	return(ax)



if __name__ == '__main__':

	steergazefile = "../Data/trout_6.feather"
	steergaze = pd.read_feather(steergazefile)
	#query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
	#data = steergaze.query(query_string).copy()

	query_string = "ID == 504 & drivingmode ==1 & th_along_midline < 1 & roadsection < 2 & T0_on_screen == 0 & confidence > .6 & on_srf == True"
	

	data = steergaze.query(query_string).copy()
	dists = np.sqrt(data.midline_vangle_dist.values ** 2 + data.midline_hangle_dist.values ** 2)
	data = data.iloc[dists<20,:]

	data = data.iloc[data.midline_cumdist.values<60,:]

	print(data.currtime.values)
	print(data.block.values)
	plt.scatter(data.midline_cumdist.values / 8, data.th_along_midline.values, alpha = .2)
	plt.ylim(0,5)
	plt.show()
	#hehe
	plt.figure()
	ax = plt.gca()
	ax = plot_screenshot(ax)

	pixels_limits_top = [1920,1080]
	xpix, ypix = data.screen_x_pix.values, data.screen_y_pix.values
	min_xpix, max_xpix = min(xpix), max(xpix)
	min_ypix, max_ypix = min(ypix), max(ypix)
	print(min_xpix, max_xpix)
	print(min_ypix, max_ypix)
	min_xpix  = 260
	max_xpix = 1670	
	max_ypix = 590
	ax.scatter(data.screen_x_pix.values, data.screen_y_pix.values, alpha = .1)
	ax.axhline(y = max_ypix, xmin = 0, xmax = 1, color = 'r')
	ax.axvline(x = min_xpix, ymin = 0, ymax = 1, color= 'r')
	ax.axvline(x = max_xpix, ymin = 0, ymax = 1, color='r')
	
	ax.set_ylim(0,pixels_limits_top[1])
	ax.set_xlim(0,pixels_limits_top[0])
	plt.show()

	#screen_x 
	#max_x = data.screen_x_pix.values

	

	
#	data.sort_values('currtime', inplace=True)

	
	
	
	
	

					