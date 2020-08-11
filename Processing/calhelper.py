import numpy as np
import pandas as pd 
from TrackMaker import TrackMaker
import sys
from scipy import spatial
import ctypes
import matplotlib
import os
sys.setrecursionlimit(10000000)

"""
Convenience library for repeated functions across scripts

"""

raw_col = 'xkcd:black'
cmap = matplotlib.cm.get_cmap('tab10')
cols = cmap([0,3])
cluster_cols = {0: cols[0], 1: cols[1], 2: '#929591'} # 2:'#f97306'
cluster_cols5 = {0: cols[0], 1: cols[1], 2:'#f97306', 3: '#ffff14',4: '#c20078',5:'#95d0fc',6:'#15b01a'}  

#https://matplotlib.org/3.1.1/tutorials/colors/colors.html

def get_midline():

	sectionsize = 10000
	TrackData = TrackMaker(sectionsize) 
	midline = TrackData[0]  
	return(midline)

def get_cumdist(midline):
	
	mid_diff = np.linalg.norm(np.diff(midline, axis=0, prepend = np.array([[0,0]])), axis = 1)
	midline_cumdist = np.cumsum(mid_diff)
	return(midline_cumdist)

def get_mirrored_traj(d):

	posx = d.posx.values
	posz = d.posz.values
	mirror = d.startingposition.values
	posx = posx * mirror
	posz = posz * mirror
	traj = np.transpose(np.array([posx, posz]))
	return(traj)

def get_tree(arr):
	
	tree = spatial.cKDTree(arr)
	return(tree)

def weighted_RGB(weights, rgbs, norm = True):
	
	n_clusts = weights.shape[0]
	rgbs = np.array(rgbs[:n_clusts])
	rgb_ws = []
	for w in weights.T:
		rgb_w = np.average(rgbs, weights=w, axis=0)		
		if norm: 			
			rgb_w /= 255
			rgb_w = np.clip(rgb_w, 0, 1) #HACK						
		rgb_ws.append(rgb_w)
	
	return(rgb_ws)

def hex_to_RGB(col_dict):
	#source: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
	rgbs = []
	for c in col_dict.values():
		if isinstance(c, str): 
			if '#' not in c: raise Exception("color needs to be hex")
			h = c.lstrip('#')
			rgbs.append(tuple( int(h[i:i+2], 16)/255 for i in (0, 2, 4)) ) 
		else:			   
			rgbs.append(tuple(c[:-1])) #already rgba
	print(rgbs)    
	return rgbs

def dictget(d, *k):
	"""Get the values corresponding to the given keys in the provided dict."""    
	if type(d) is not dict: raise Exception('first argument is type ' + str(type(d)) + ', should be a dict')
	return (d[i] for i in k) 

def limit_to_data(ax, data, margin):    

	ax.set_xlim(min(data[:,0]) - margin[0], max(data[:,0]) + margin[0])       
	ax.set_ylim(min(data[:,1]) - margin[1], max(data[:,1]) + margin[1])   
	return(ax)

def norm_to_pixels(x, y):

	"""converts normalised units to pixels of current display"""
	user32 = ctypes.windll.user32
	width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
	y = 1-y #flip y.
	pixels = x * width, y * height
	return(pixels)

def move_figure(f, x, y):

	"""Move figure's upper left corner to pixel (x, y). Position given in normalised units. Bottom left is 0,0.
	
	based on cxrodgers answer on https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
	"""
	x, y = norm_to_pixels(x, y)

	backend = matplotlib.get_backend()
	if backend == 'TkAgg':
		f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
	elif backend == 'WXAgg':
		f.canvas.manager.window.SetPosition((x, y))
	else:
		# This works for QT and GTK
		# You can also use window.setGeometry
		f.canvas.manager.window.move(x, y)


def check_exist(filepath):

	if ':' not in filepath: filepath = os.path.join(os.getcwd(), filepath)
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