import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linmix as lm
from scipy.stats import norm, t, pearsonr, spearmanr, linregress, stats
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import polyfit
import calhelper as ch
import matplotlib.cm as cm
from scipy.odr import Model, Data, ODR
from memoize import memoize


def load_data(rerun = True, plot = False):
	
	if rerun:
		res = lm.main(plot=plot)
	else:
		#res = pd.read_csv("linmix_res_4.csv")   
		#res = pd.read_csv("linmix_by_block_3.csv")
		#res = pd.read_csv("linmix_matched.csv")
		res = pd.read_csv("linmix_d1_3.csv")
		#res = res.query('ID != 4').copy()
		res = res.query('dataset == 2').copy()
	return(res)


def abline(slope, intercept, axes = None):
	"""Plot a line from slope and intercept"""
	if axes == None: axes = plt.gca()
	ylim = axes.get_ylim()
	 # print(ylim)
	x_vals = np.array(axes.get_xlim())
	y_vals = intercept + slope * x_vals
	axes.plot(x_vals, y_vals, '--', color = 'gray')
	#axes.set_ylim(ylim[0], ylim[1] + 1)

def orthoregress(x, y):
	"""Perform an Orthogonal Distance Regression on the given data,
	using the same interface as the standard scipy.stats.linregress function.
	Arguments:
	x: x data
	y: y data
	Returns:
	[m, c, nan, nan, nan]
	Uses standard ordinary least squares to estimate the starting parameters
	then uses the scipy.odr interface to the ODRPACK Fortran code to do the
	orthogonal distance calculations.

	lifted from: http://blog.rtwilson.com/orthogonal-distance-regression-in-python/
	"""
	linreg = linregress(x, y)
	mod = Model(f)
	dat = Data(x, y)
	od = ODR(dat, mod, beta0=linreg[0:2])
	out = od.run()
	return list(out.beta)
def f(p, x):
	"""Basic linear regression 'model' for use with ODR"""
	return (p[0] * x) + p[1]

@memoize
def empirical_means():
	
	steergazefile = "../Data/trout_subset_3.feather"
	steergaze = pd.read_feather(steergazefile)
	query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
	data = steergaze.query(query_string).copy()
	dists = np.sqrt(data.midline_vangle_dist.values ** 2 + data.midline_hangle_dist.values ** 2)
	data = data.iloc[dists<20,:]
	data.loc[data.ID == 203, 'ID'] = 503  

	out = data.groupby(['ID','drivingmode']).th_along_midline.mean()	
	return(out)

def weighted_means(res, pooled, ax):

	"""plots estimated mean decomposition"""


	#rgbas = cmap([6,1,2,4])
	cmap = cm.get_cmap('tab10')
	titles = {0: 'Active', 1: 'Replay', 2:'Stock'}
	active_color = tuple(np.array([3, 238, 255]) / 255)
	replay_color = cmap([1])[0]
	print(replay_color)
	stock_color = tuple(np.array([255, 3, 121]) / 255)
	cmaps = {0: active_color, 1: replay_color, 2: stock_color}

	#TODO: need to normalise the weights before adding up the composed means
	res = res.query("clust_n < 2")
	pooled = pooled.reset_index()	

	dm_nudge = .175
	size_scale = 10
	

	IDs = list(set(res.ID.values))

	#three contrasts. #Passive-Active, Stock-Active, Stock-Passive
	#three clusters
	partial = np.empty([len(IDs),3,2])
	overall = np.zeros([len(IDs),3])	
	for (pi, dm), d in res.groupby(['ID','drivingmode']):

		mns, wts, cis = d['mean'].values, d['weight'].values, d['clust_n'].values		
		wts /= np.sum(wts)
		wt_mns = np.multiply(mns,wts)

		ppidx = IDs.index(pi)	
				
		for m, ci, w in zip(mns, cis, wts):
			#print(dm)
			ax.plot(ppidx + (dm*dm_nudge), m, marker = 'o', markerfacecolor = cmaps[dm], color = ch.cluster_cols[ci], markersize = 5) #ms = w*size_scale)	
		
		partial[ppidx, dm, :] = mns		
		overall[ppidx, dm] = np.sum(wt_mns)
				
	#plot contrasts
	
	

	for i in range(len(IDs)):			

		ppemp = pooled.query(f'ID == {IDs[i]}').th_along_midline.values				
		#emp_con = [ppemp[1]-ppemp[0],ppemp[2]-ppemp[0],ppemp[2]-ppemp[1]]		
		for co in [0,1,2]: 				

			full = overall[i,co]			
			#ax.plot(full, i + co*dm_nudge, 'o', mfc = (1,1,1), mec = 'k', alpha = 1)#, ms = .5 * size_scale)			
			ax.plot(i + co*dm_nudge, full, 'o', mfc = "white", mec = cmaps[co], alpha = 1)#, ms = .5 * size_scale)			
			#empirical
			ax.hlines(y = ppemp[co], xmin = (i + co*dm_nudge) - .07, xmax = (i + co*dm_nudge) + .07, linestyle = '-', color = (.0,.0,.0), alpha = 1, zorder = 3)#, ms = .5 * size_scale)
			
			"""
			#for vertical segments.
			for cl in [0, 1]:
					ax.vlines(x = i + co*dm_nudge, ymin = partial[i,co,cl], ymax = full, color = ch.cluster_cols[cl], linestyle = '-', alpha = .7, zorder = -1)
			"""

		
		#for horizontal segmentss
		x = i + np.array([0,1,2])*dm_nudge
		ax.plot(x, overall[i,:], color = (.8,.8,.8), linestyle = '-', zorder = -1)
		for cl in [0,1]:
			
			ax.plot(x, partial[i,:,cl], color = ch.cluster_cols[cl], linestyle = '-', alpha = .7, zorder = -1)
		
			
					
			#ax[co].set_title(titles[co])
	ticks = list(range(0,11))
	ax.set_xticks(ticks)
	tick_str = [str(t+1) for t in ticks]
	ax.set_xticklabels(tick_str)
	ticks = list(range(0,6))
	ax.set_yticks(ticks)
	tick_str = [str(t) for t in ticks]
	ax.set_yticklabels(tick_str)
	for y in ticks: ax.axhline(y, xmin = 0, xmax = 1, color = (.95,.95,.95), linestyle = '--', zorder = -2)
	ax.set_ylim(0,4.5)
		
	#ax.invert_xaxis()
	ax.set_ylabel("Time Headway (s)")
	ax.set_xlabel("Participant")

		
	legend1 = [Line2D([0], [0], marker='o', color = ch.cluster_cols[1], markerfacecolor= (.8,.8,.8), label='EF',
							alpha =1, markersize = 5),
				Line2D([0], [0], marker='o', color = ch.cluster_cols[0], markerfacecolor=(.8,.8,.8), label='GF',
							alpha =1, markersize = 5)]

	#legend2 = []
	
	legend3 = [Line2D([0], [0], marker='o', color = "w", markerfacecolor="white", markeredgecolor = cmaps[0], label='Manual',
							alpha =1),
				Line2D([0], [0], marker='o', color = "w", markerfacecolor="white", markeredgecolor = cmaps[1], label='Replay',
							alpha =1),
				Line2D([0], [0], marker='o', color = "w", markerfacecolor="white", markeredgecolor = cmaps[2], label='Stock',
							alpha =1),
							Line2D([0], [0], color = (.0,.0,.0), label='Empirical Mean',
							alpha =1)]
	

	first_legend = plt.legend(handles=legend1, loc = 4, fontsize = 8, frameon = True, title =  "Decomposed Means")
	ax.add_artist(first_legend)
	#leg2 = plt.legend(handles=legend2, loc = 8, fontsize = 8, frameon = False)
	#ax.add_artist(leg2)
	ax.legend(handles=legend3, loc = 3, fontsize = 8, frameon = True, ncol = 4, title = "Composed Means                                                       ")

	return overall, partial


def weighted_differences(res, pooled, ax):
	"""plots contrast decomposition """

	res = res.query("clust_n < 2")
	pooled = pooled.reset_index()	
	

	IDs = list(set(res.ID.values))

	#three contrasts. #Passive-Active, Stock-Active, Stock-Passive
	#two clusters
	partial = np.empty([len(IDs),3,2])
	overall = np.zeros([len(IDs),3])	
	for (pi, ci), d in res.groupby(['ID','clust_n']):


		mns, wts = d['mean'].values, d['weight'].values
		wt_mns = np.multiply(mns,wts)
		#wt_mns = mns
		
		pasact = wt_mns[1] - wt_mns[0]
		stoact = wt_mns[2] - wt_mns[0]
		stopas = wt_mns[2] - wt_mns[1]

		print(pi, ci)
		print("model_mns: ", mns)
		print("model_wts: ", wts)
		print("wt_mns: ", wt_mns)
		print("Replay - Active: ", pasact)
		print("Stock - Active: ", stoact)
		print("Stock - Passive: ", stopas)
		print("\n")

		ppidx = IDs.index(pi)		
		partial[ppidx, 0, ci] = pasact
		partial[ppidx, 1, ci] = stoact
		partial[ppidx, 2, ci] = stopas

		overall[ppidx, 0] = overall[ppidx, 0] + pasact
		overall[ppidx, 1] = overall[ppidx, 1] + stoact
		overall[ppidx, 2] = overall[ppidx, 2] + stopas
		
		
	#plot contrasts
	titles = {0: 'Replay-Active', 1: 'Stock-Active', 2:'Stock-Replay'}
	for i in range(len(IDs)):			

		ppemp = pooled.query(f'ID == {IDs[i]}').th_along_midline.values				
		emp_con = [ppemp[1]-ppemp[0],ppemp[2]-ppemp[0],ppemp[2]-ppemp[1]]
		for co in [0,1,2]: 
			
			for cl in [0,1]:				
				part = partial[i, co, cl]	
				ax[co].plot(part, i, 'o', color = ch.cluster_cols[cl], alpha = .8)

			full = overall[i,co]
			ax[co].plot(full, i, 'o', color = 'k', alpha = .8)
			ax[co].axvline(0, ymin = 0, ymax = 1, color = (.5,.5,.5), linestyle = '--', zorder = -1)
			ax[co].set_title(titles[co])
			ticks = list(range(0,11))
			ax[co].set_yticks(ticks)
			tick_str = [str(t+1) for t in ticks]
			ax[co].set_yticklabels(tick_str)
			ax[co].set_xticks(list(np.arange(-.2,.6,.2)))

			#empirical
			ax[co].plot(emp_con[co], i, 'o', color = (.5,.5,.5), alpha = .8)
			ax[0].invert_yaxis()



def publication_plot(res):

	lower, upper, steps = 0, 7, 500
	binsize = (upper-lower) / steps
	x = np.linspace(lower, upper, steps)


	fig_cm = np.array([18,15])
	fig_inc = fig_cm /2.54 
	fig, axes = plt.subplots(5,2, constrained_layout = True, figsize = fig_inc, sharex = False, sharey = False)
	
	for ci, d in res.groupby(['clust_n']):
	
		if ci == 2: continue
		for dm, dd in d.groupby(['drivingmode']):

			means, weights = [], []
			for pi, ddd in dd.groupby(['drivingmode','ID']):

				print((ci,dm,pi))
				
				mn, var, wt = ddd['mean'].values, ddd['variances'].values, ddd['weight'].values
					
				dens = norm.pdf(x, mn, np.sqrt(var)) * binsize * wt
				axes[dm,ci].plot(x, dens, color = ch.cluster_cols[ci], alpha = .25)		

				means.append(mn)
				weights.append(wt)
			
			for i, vals in enumerate([means, weights]):
				vals = np.array(vals)
				avg = np.mean(vals)
				axes[3+i,ci].plot(avg,dm*-1, 'o', c = ch.cluster_cols[ci])
				axes[3+i,ci].hlines(dm*-1, xmin = min(vals), xmax = max(vals), color = ch.cluster_cols[ci])
		
	#plot means
	for i in [0,1]:

		axes[-2,i].set_ylim(-2.5,.5)
		axes[-2,i].set_xlim(0,7)
		axes[-1,i].set_ylim(-2.5,.5)
		axes[-1,i].set_xlim(0,1)

	plt.show()

def plot_corr(res, pairs = [[0,1],[0,2],[1,2]]):

	"""plots multiplot inter-individual correlations across driving mode for time headway and weights"""

	#time headway, weights, variances
	

	#titles = ['Time Headway Mean', 'Proportions']
	titles = ['Time Headway Mean']

	fig, ax = plt.subplots(2,len(pairs), figsize=(15,15),  num = "correlations")


	names = {0: 'active', 1: 'replay', 2: 'stock'}

	for pi, drives in enumerate(pairs):
		
		xname = names[drives[0]]
		yname = names[drives[1]]
		xlabs = [f'{xname} mean th (s)', f'{xname} weight (%)']
		ylabs = [f'{yname} mean th (s)', f'{yname} weight (%)']
		
		datacolumns = ['mean']
		noise = max(res.clust_n.values)

		for ci, (clust, d) in enumerate(res.groupby(['clust_n'])):
			
			if clust == noise: continue
			
			active = d.query('drivingmode == {}'.format(drives[0])).copy()
			active.sort_values('ID',inplace=True)
			passive = d.query('drivingmode == {}'.format(drives[1])).copy()
			passive.sort_values('ID',inplace=True)

			for di, dc in enumerate(datacolumns):
				act = np.array(active.loc[:,dc])
				pas = np.array(passive.loc[:,dc])
				dataset = active.dataset.values

				r,p = pearsonr(act,pas)
				#r,p = spearmanr(act,pas)

				#itc, slo = polyfit(act, pas, 1)
				beta = orthoregress(act, pas)

				axes = ax[ci, pi]

				
				axes.text(.7, .2, 'r: ' + '% .2g'%r, transform = axes.transAxes)
				axes.text(.7, .1, 'p: ' + '% .2g'%p, transform = axes.transAxes)
				axes.plot(act, beta[1] + beta[0] * act, 'k-')
				#axes.plot(act, itc + slo * act, 'g-')
				abline(1, 0, axes)
				axes.set(xlabel = xlabs[di], ylabel = ylabs[di])

				mcol = {1: ch.cluster_cols[clust], 2:(0,0,0)}
				for a, p, d in zip(act, pas, dataset):

					axes.plot(a, p, 'o', markeredgecolor = mcol[d], color = ch.cluster_cols[clust])
		
		for a, ct in zip(ax[0], titles):
			a.set_title(ct)
		
		rowtls = ['GF','Entry','Exit']
		for a, rt in zip(ax[:,0], rowtls):
			a.annotate(rt, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0),
					xycoords=a.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center', rotation = 90)
				
	plt.savefig(f'corr_linmix_{xname}_{yname}.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.savefig(f'corr_linmix_{xname}_{yname}.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show() 


def plot_hists(res, pairs = [[0,1],[0,2],[1,2]]):

	"""plots multiplot inter-individual correlations across driving mode for time headway and weights"""

	#time headway, weights, variances
	
	res = res.query('clust_n == 0')
	#titles = ['Time Headway Mean', 'Proportions']
	titles = ['Time Headway Mean']

	fig, ax = plt.subplots(1,len(pairs), figsize=(15,15),  num = "correlations")


	names = {0: 'active', 1: 'replay', 2: 'stock'}

	for pi, drives in enumerate(pairs):
		
		xname = names[drives[0]]
		yname = names[drives[1]]
		xlabs = [f'{xname} mean th (s)', f'{xname} weight (%)']
		ylabs = [f'{yname} mean th (s)', f'{yname} weight (%)']
		
		datacolumns = ['mean']
												
		active = res.query('drivingmode == {}'.format(drives[0])).copy()
		active.sort_values('ID',inplace=True)
		passive = res.query('drivingmode == {}'.format(drives[1])).copy()
		passive.sort_values('ID',inplace=True)

		for di, dc in enumerate(datacolumns):
			act = np.array(active.loc[:,dc])
			pas = np.array(passive.loc[:,dc])
			dataset = active.dataset.values

			diff = pas - act
				#r,p = spearmanr(act,pas)
			ax[pi].hist(diff)
				
	plt.show()


def CIs(d, ci = .99):

	m = np.nanmean(d)
	n = len(d)
	sd = np.sqrt (np.sum( (d - m)**2 ) / (n-1))
	se = sd / np.sqrt(n)
	cis = se * t.ppf( (1+ci) / 2, n-1)  
		
	return(m, cis)


def plot_within(res, drives = [0,1]):
	"""plots within participant differences"""

	#time headway, weights, variances
	fig, ax = plt.subplots(1,3, figsize=(10,6), num = "within participant differences")

	titles = ['Time Headway Mean','Time Headway Mode', 'Proportions']
	
	ylabs = ['Passive - Active mean th (s)','Passive - Active mode (s)', 'Passive - Active weight (%)']
	
	datacolumns = ['mean','mode','weight']
	noise = max(res.clust_n.values)

	for i, (clust, d) in enumerate(res.groupby(['clust_n'])):
		
		if clust == noise: continue
		
		active = d.query('drivingmode == {}'.format(drives[0])).copy()
		active.sort_values('ID',inplace=True)
		passive = d.query('drivingmode == {}'.format(drives[1])).copy()
		passive.sort_values('ID',inplace=True)


		for di, dc in enumerate(datacolumns):
			act = np.array(active.loc[:,dc])
			pas = np.array(passive.loc[:,dc])
			dataset = active.dataset.values

			diff = pas-act
			m, ci= CIs(diff, ci = .95)

			axes = ax[di]

			t, p = stats.ttest_1samp(diff,0)
						
			axes.errorbar((clust*.3) + .5, m, yerr = ci, c = ch.cluster_cols[clust])
			axes.plot((clust*.3) + .5, m, c = ch.cluster_cols[clust], marker = 'o')
			axes.set_title(titles[di])
			#ax.flat[i].set(xlabel = 'Road Section', ylabel = ylabs[i])
			axes.text((clust*.5 + .1), .2, 't: ' + '% .2g'%t, transform = axes.transAxes)
			axes.text((clust*.5 + .1), .15, 'p: ' + '% .2g'%t, transform = axes.transAxes)
			axes.set_xticks([]) 
			axes.set_xticklabels([''])  
			axes.axhline(y=0.0, color=(.4,.4,.4), linestyle='--')
	
	legend_elements = [Line2D([0], [0], color = 'xkcd:blue', lw=4, label='GF'),
						Line2D([0], [0], color = 'xkcd:red', lw=4, label='Entry')
						#  Line2D([0], [0], color = 'xkcd:orange', lw=4, label='Exit')                       
					 ]
	ax.flat[-1].legend(handles=legend_elements, loc = [.6,.8])
	#plt.savefig('withindiffs_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show()

def plot_blocks(res):
	"""plots a mean th across blocks etc"""
	
	markers = {0: 'o', 1: 's', 2:'*'}
	al = {0: 1, 1: .5, 2:.5}
	
	fig, ax = plt.subplots(1,2, figsize=(10,6),  num = "per block")

	titles = ['GF','LAF']
	ylabs = ['mean th (s)']
	
	print(res.block.values)
	
	noise = max(res.clust_n.values)
	for g, d in res.groupby(['drivingmode','block','clust_n']):        
		drive, block, clust_n = g[0], g[1], g[2]
		if clust_n == noise:continue #noise
		datlist = d['mean'].values        

		clust_n = d.clust_n.values[0]         
		
		i = clust_n

		#print(titles[i])
		m, ci= CIs(datlist, ci = .95)
		#print('mean', m)
		#print('CI', ci)
		ax.flat[i].errorbar((block*.1)+(drive*.8), m, yerr = ci, c = ch.cluster_cols[clust_n], alpha = al[drive])
		ax.flat[i].plot((block*.1)+(drive*.8), m, c = ch.cluster_cols[clust_n], marker = markers[drive], alpha = al[drive])
		ax.flat[i].set_title(titles[i])
		#ax.flat[i].set(xlabel = 'Road Section', ylabel = ylabs[i])
		#ax.flat[i].set_xticks([]) 
		#ax.flat[i].set_xticklabels([''])        
	
	legend_elements = [Line2D([0], [0], color = 'xkcd:blue', lw=4, label='GF'),
						Line2D([0], [0], color = 'xkcd:red', lw=4, label='Entry'),
						#Line2D([0], [0], color = 'xkcd:orange', lw=4, label='Exit'),
						Line2D([0], [0], marker='o', color='w', label='Active',
							alpha =1, markerfacecolor='k'),
						Line2D([0], [0], marker='s', color='w', label='Passive',
							alpha =.5, markerfacecolor='k'),
							Line2D([0], [0], marker='*', color='w', label='Stock',
							alpha =.5, markerfacecolor='k')
					 ]
	ax.flat[-1].legend(handles=legend_elements, loc = [.4,.0])
	#plt.savefig('cluster_pointranges_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show()

def model_error(res, pooled, composed_means):

	res = res.query("clust_n == 2") #noise cluster
	pooled = pooled.reset_index()	

	IDs = list(set(res.ID.values))
	error = []
	for ix, i in enumerate(IDs):
		for dm in [0,1,2]:
			noiseweight = res.query(f'ID == {i} & drivingmode == {dm}').weight.values[0]
			cmean = composed_means[ix,dm]
			empmean = pooled.query(f'ID == {i} & drivingmode == {dm}').th_along_midline.values[0]

			err = cmean - empmean
			error.append([err, abs(err), noiseweight])

	error = np.array(error)
	print(np.average(error[:,1]))
	print(np.sqrt(np.var(error[:,1])))
		
	#print(pearsonr(error[:,0],error[:,2]))
	#print(pearsonr(error[:,1],error[:,2]))
	#print(spearmanr(error[:,0],error[:,2]))
	#print(spearmanr(error[:,1],error[:,2]))			


def plot_pointranges(res):
	"""plots a multiplot cluster size, mean_th etc"""
	
	markers = {0: 'o', 1: 's', 2:'*'}
	al = {0: 1, 1: .5, 2:.5}
	
	titles = ['Mean Time Headway', 'Cluster Proportions']

	fig, ax = plt.subplots(1,len(titles), figsize=(10,6),  num = "point estimates")
	ylabs = ['mean th (s)', 'mean weight (%)']
	
	noise = max(res.clust_n.values)
	for g, d in res.groupby(['drivingmode','clust_n']):        
		drive, clust_n = g[0], g[1]
		if clust_n == noise:continue #noise
		datlist = [d['mean'].values,
		d['weight'].values]

		clust_n = d.clust_n.values[0]         
		
		for i, arr in enumerate(datlist):

			#print(titles[i])
			m, ci= CIs(arr, ci = .95)

			print(g)
			print(titles[i])
			print('mean', m)
			print('CI', m - ci, m + ci)
			print("\n")

			ax.flat[i].errorbar((clust_n*.3)+(drive*.1), m, yerr = ci, c = ch.cluster_cols[clust_n], alpha = al[drive])
			ax.flat[i].plot((clust_n*.3)+(drive*.1), m, c = ch.cluster_cols[clust_n], marker = markers[drive], alpha = al[drive])
			ax.flat[i].set_title(titles[0])
			ax.flat[i].set(xlabel = 'GF                               Entry', ylabel = ylabs[i])
			ax.flat[i].set_xticks([]) 
			ax.flat[i].set_xticklabels([''])        
	
	legend_elements = [Line2D([0], [0], color = 'xkcd:blue', lw=4, label='GF'),
						Line2D([0], [0], color = 'xkcd:red', lw=4, label='Entry'),
						#Line2D([0], [0], color = 'xkcd:orange', lw=4, label='Exit'),
						Line2D([0], [0], marker='o', color='w', label='Active',
							alpha =1, markerfacecolor='k'),
						Line2D([0], [0], marker='s', color='w', label='Passive',
							alpha =.5, markerfacecolor='k'),
							Line2D([0], [0], marker='*', color='w', label='Stock',
							alpha =.5, markerfacecolor='k')
					 ]
	ax.flat[-1].legend(handles=legend_elements, loc = [.6,.6])
	ax.flat[0].set_ylim(ymin=1.5, ymax = 4)
	#ax.flat[-1].set_ylim(ymin = 0, ymax = 1)
	plt.savefig('cluster_pointranges_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	plt.show()

res = load_data(rerun = False, plot = False)
#plot analytic marginal cluster distributions in four subplots

#publication_plot(res)
pooled_mns = empirical_means()




	#add_vlines(f4.axes[1], means)
fig_cm = np.array([18,8])
fig_inc = fig_cm /2.54 

fig,ax = plt.subplots(1,1, constrained_layout = True, sharex= True, sharey = True, num = "Means", figsize = fig_inc)
overall, partial = weighted_means(res, pooled_mns, ax)
ch.move_figure(fig, .6,1)
plt.savefig("decomposed_means_4.png", format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')


#model error
model_error(res, pooled_mns, overall)	

"""
fig2,ax2 = plt.subplots(1,3, constrained_layout = True, sharex= True, sharey = True, num = "Contrasts",  figsize = (6,4))
weighted_differences(res, pooled_mns, ax2)
ch.move_figure(fig2, .6,.5)
plt.savefig("decomposed_contrasts.png", format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
"""

plt.show()

#drives = [0,1]

#plot_pointranges(res)

#pairs = [[0,1],[0,2],[1,2]]

#plot_corr(res, pairs)

#plot_hists(res, pairs)

#plot_within(res, drives)

#plot_blocks(res)

#check_pp_effect(res)