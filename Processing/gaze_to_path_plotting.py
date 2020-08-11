import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os
import pandas as pd
from file_methods import *
from pprint import pprint

import drivinglab_projection as dp
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from memoize import memoize
from matplotlib.lines import Line2D

def plot_distributions(df, ax, cutoff = 5):

	#th = df.th_along_midline.values
	x = np.linspace(0, 2, 500)	

	cols = cm.Set1(range(4))	
	#cnames = {0: 'Attract\nNarrow', 1:'Attract\nWide', 2:'Repel\nNarrow',3:'Repel\nWide'}
	cnames = {0: 'Active', 1: 'Replay', 2: 'Stock'}
	
	#for g, d in df.groupby('condition'):
	for g, d in df.groupby('drivingmode'):
				
		g = int(g)
		gtp = d.gazetopath_metres.values
		gtp = gtp[gtp<cutoff]
		density = gaussian_kde(gtp)		
		ax.plot(x,density(x), color = cols[g]) #row=0, col=0		

		mn = np.mean(gtp)
		#ax.axvline(ymin = 0, ymax = .2, x = mn, c = cols[g], linestyle = '-')
		med = np.median(gtp)
		#ax.axvline(ymin = 0, ymax = .05, x = med, c = cols[g], linestyle = ':')
		print("condition: ", g)
		print("mean: ", mn)
		print("med: ", med)

		for p, dd in d.groupby('ID'):

			p_gtp = dd.gazetopath_metres.values
			p_gtp = p_gtp[p_gtp<cutoff]
			ax.plot(np.median(p_gtp), -.1 + (g*-.1), '.', c = cols[g], alpha = .5)

	ax.set_xlabel('Gaze to Path (m)', fontsize = 12)    
	ax.set_ylabel('Density', fontsize = 12)    

	legend_elements = []
	for c in range(3):
		legend_elements.append(Line2D([0], [0], color = cols[c], lw=4, label=cnames[c]))
	
	ax.legend(handles=legend_elements, loc = [.7,.6])

if __name__ == '__main__':

	steergazefile = "../Data/trout_6.feather"
	steergaze = pd.read_feather(steergazefile)
	
	#dm_names = {0: 'Active', 1: 'Replay', 2: 'Stock'}

	#for dm in [0,1,2]:
	dm_names = {0: 'AttractNarrow', 1:'AttractWide', 2:'RepelNarrow',3:'RepelWide'}
	for c in [0,1,2,3]:
		
		#query_string = f"drivingmode ==  {dm} & (T0_on_screen ==1 | roadsection == 2) & confidence > .6 & on_srf == True & dataset == 2"
		query_string = f"condition ==  {c} & (T0_on_screen ==1 | roadsection == 2) & confidence > .6 & on_srf == True & dataset == 2"
		data = steergaze.query(query_string).copy()

		fig, axes = plt.subplots(1,1)

		plot_distributions(data,axes)

		fig.suptitle(dm_names[c], fontsize=16)
		fname = 'slalom_gaze_to_path_metres' + dm_names[c] +'.png'
		plt.savefig(fname, format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
		plt.show()


			
	
	
	
	
	

					