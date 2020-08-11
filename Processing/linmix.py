import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, gaussian_kde
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.special import softmax, logsumexp
import calhelper as ch
import scipy.ndimage
from scipy.interpolate import interp1d
from drivinglab_projection import plot_perspective_view as perspective
from memoize import memoize
from matplotlib.lines import Line2D


"""
mixture linear regression model with four models.
mu = intercept
gf: th ~ N(mu0, sd0)
entry: th ~ N(mu1 + beta1*time, sd1)
exit : th ~ N(mu2 + beta2*time, sd2)
noise: th ~ N(mu3, sd3)

"""

def entropy_scores(fit):

	"""shannons entropy"""
	resps = fit['resps']
	entropy = np.array([-np.sum(r * np.log(r))
	for r in resps.T])	
   # print(entropy.shape)
	return(np.mean(entropy))

def load_cluster_fits():

	res = pd.read_csv("gmm_res.csv")   
	return(res)

def load_data(file):

	#df = pd.read_feather('../Data/trout_2.feather')
	df = pd.read_feather(file)
	#query_string = "drivingmode < 3 & confidence > .8"	
	#query_string = "drivingmode < 3 & confidence > .6 & dataset == 1 & on_srf == True"
	query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
	df = df.query(query_string).copy()
	#correct for typo
	df.loc[df.ID == 203, 'ID'] = 503  

	return(df)

def predict_trial(res):
	 
	"""should be one participant"""

	n_obs = 2000
	X = []
	#columns = ['clust_n','clust_type','clust_veh_mean','clust_th_mean','clust_veh_var','clust_th_var','clust_prop','sil_score','entropy','roadsection','drivingmode','ID']
	for _, row in res.iterrows():

		if row.clust_type == "noise": continue
		m = [row.clust_veh_mean, row.clust_th_mean]
		cov = [[row.clust_veh_var, row.clust_cov],
				[row.clust_cov, row.clust_th_var]]
		prop = row.clust_prop
		
		X.append(multivariate_normal.rvs(mean=m, cov=cov, size=int(prop*n_obs)))

	X_predict = np.concatenate(X)
	return(X_predict, X)

def linfits(clusters):

	regs = []
	model_types = ['empty','slope','slope','empty']
	for i, c in enumerate(clusters):
		
		if i == 3: continue
		print("cluster: ", i)
		#X = np.array([intercept, c[:,0]]).T #predictor array
		model = model_types[i]
		y = c[:,1]
		if model == 'empty':
			X = np.ones(c.shape[0]).reshape(-1,1) #slope of predictor acts as intercept.           
			reg = LinearRegression(fit_intercept=False).fit(X, y)

			#preds = np.dot(X, reg.coef_) + reg.intercept_
			preds = reg.predict(X)
			print(np.std(preds-y))
		elif model == 'slope':
			X = c[:,0].reshape(-1,1) #need intercept and predictor
			reg = LinearRegression(fit_intercept=True).fit(X,c[:,1])        
			preds = reg.predict(X)
			print(np.std(preds-y))
		else:
			raise Exception('unrecognised model type')
	  
		print(reg.coef_)
		print(reg.intercept_)
		print(reg.get_params())
		regs.append(reg)

	return(regs)

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def weights_over_time(trials, smooth = True, plot = True, ax = None):

	"""returns smoothed coefficients per time point"""

	cd = trials[0][0]	
	rlist = np.array([t[2] for t in trials]) #Ntrials x Nclust x samples	
	if smooth: rlist = np.array([scipy.ndimage.gaussian_filter1d(r, 10) for rs in rlist for r in rs]).reshape(rlist.shape) #gauss filter with 10cm step
		
	w_t = [np.nanmean(rlist[:,c,:], axis = 0) for c in range(rlist.shape[1])] ##average across trials weight over time, w_t, for each cluster 			
	w_t /= np.sum(w_t, axis = 0) #sum to one.	

	if plot:
		zord = {0: 3, 1: 2, 2: 1}
		for i, w in enumerate(w_t):
			if ax == None: ax = plt.gca()			
			ax.plot(cd, w, '.', color = ch.cluster_cols[i], alpha = .1, zorder = zord[i])

	#print(min(cd), max(cd))
	
	#return(w_t)
	return(ax)

@memoize
def map_trials(fit):

	"""
	for each block return a list of trials of equal length.
	Returns Trials: list of [cumdist, th, resps] of length Ntrials, where resps in a list of length n_clust
	"""
	data, models, resps = ch.dictget(fit, 'data', 'models', 'resps')
	cumdist = data[:,0]
	#cd = np.arange(min(cumdist), max(cumdist), step = .01)
	cd = np.arange(0, 20, step = .01) #constant array throughout all trials.

	#get array of trial indexes	
	cd_diff = np.diff(cumdist)	
	starts = np.where(cd_diff < 0)[0] + 1
	starts = np.insert(starts, 0,0)
	starts = np.append(starts, len(cumdist)-1)	

	trial_data = []
	trial_resps = []
	for i, st in enumerate(starts):
		if i == (len(starts)-1): continue #skip last index.
		trial_data.append(data[st:starts[i+1],:])
		trial_resps.append(resps.T[st:starts[i+1],:])		

	def interp(x, y, z):
		interp = interp1d(x, y, bounds_error = False, fill_value = np.nan)
		return(interp(z))

	interp_trials = []
	for trial, resp in zip(trial_data, trial_resps):
		th = interp(trial[:,0], trial[:,1], cd)		
		res = [interp(trial[:,0], r, cd) for r in resp.T]		
		out = [cd, th, res]
		interp_trials.append(out)
		
	return(interp_trials)

def plot_density(fit, trials, w_t):
	
	models = fit['models']	
	th = fit['data'][:,1]
	
	cd = trials[0][0] #fit['data'][:,0]#
	rlist = np.array([t[2] for t in trials]) #Ntrials x Nclust x samples					]
	#th_mat = np.array([t[1] for t in trials]) #ntrials x n

	#empirical density
	density = gaussian_kde(th)
	x = np.linspace(0, 7, 500)
	
	ax = plt.gcf().subplots(1,2)
	ax[0].plot(x,density(x), ch.raw_col) #row=0, col=0
	ax[1].plot(x,density(x), ch.raw_col) #row=0, col=0
	plt.xlabel('gaze time headway')    

	ntrials, nmodels, nsamples, ndens = len(trials), len(models), len(cd), len(x)
	
	totaldens = np.zeros([ndens])        
	modeldens = np.zeros([nmodels, nsamples, ndens])	

	for i, (m, ws) in enumerate(zip(models, w_t)):
		
		preds = m.predict(cd) #for each model predict the y value.                 		
		
		for t, (p, w) in enumerate(zip(preds, ws)): #for each t get the scaled density distribution
			dns = norm.pdf(x, p, m.spread) #evaluate prob density for empirical th range
			#print(w)
			dns *= w		
			#print(dns)
		
			modeldens[i, t, :] = dns / nsamples     
						
			"""
			plt.cla()
			plt.plot(x, dns, '-', color = ch.cluster_cols[i])
			plt.title("At cumdist: " + str(cumdist[t]))
			plt.draw()
			plt.pause(.016)
			"""			   		
		meandens = np.nansum(modeldens[i, :, :], axis = 0)				

		ax[1].plot(x, meandens, '-', color = ch.cluster_cols[i])
		#plt.plot(x, meandens, '-', color = ch.cluster_cols[i])
				
		#add to total dens
		totaldens += meandens

	ax[0].plot(x, totaldens, 'xkcd:magenta')    
	
def plot_average_dens(fit, dens_mat, means, variances, weights, ax = None):
	"""dens_mat has shape (ntrials, nmodels, nsamples, ndens). Each trial matrix sums to one"""

	ntrials, nmodels, nsamples, ndens = dens_mat.shape
	th = fit['data'][:,1]
	lower, upper, steps = 0, 7, 500
	x = np.linspace(lower, upper, steps)
	binsize = (upper-lower) / steps
	density = gaussian_kde(th)
	pdf_x = density.pdf(x) * binsize

	if ax == None:	ax = plt.gcf().subplots(1,2)
	ax[0].plot(x,pdf_x, ch.raw_col) #row=0, col=0
	ax[1].plot(x,pdf_x, ch.raw_col, alpha = .8) #row=0, col=0
	
	for tick in list(range(0,8)): ax[1].axvline(tick,0, 1, color = (.9,.9,.9), linestyle = '--',zorder = -1) #row=0, col=0
	
	#plt.xlabel('gaze time headway')    

	#get individual model density marginalised across time for each trial
	trial_clust_dens = np.array([np.nanmean(m, axis = 0) for t in dens_mat for m in t]).reshape(ntrials, nmodels, ndens)

	#get mean across trials.
	mean_clust = [np.nanmean(trial_clust_dens[:, m, :],axis = 0) for m in range(nmodels)]
	
	zord = {0: 3, 1: 2, 2: 1}					
	for i, m in enumerate(mean_clust):
		mean, var, wt = means[i], variances[i], weights[i]
		
#		binsize = (upper-lower) / steps
		#dens = norm.pdf(x, mean, np.sqrt(var)) * binsize * wt
		#ax[1].plot(x, dens, color = ch.cluster_cols[i], alpha = .5)				

		
		ax[1].plot(x, m, color = ch.cluster_cols[i], zorder = zord[i], alpha = .8)
		if i < 2: ax[1].axvline(mean, 0,1, color = ch.cluster_cols[i], linestyle = ':')
	
	
	totaldens = np.sum(np.array(mean_clust), axis = 0)
	ax[0].plot(x, totaldens, color = 'xkcd:black', linestyle = "dotted")  

	return(ax)	

@memoize
def estimate_density_matrix(fit, trials, plot = False):
	
	models = fit['models']	
	th = fit['data'][:,1]
	
	cd = trials[0][0]
	rlist = [t[2] for t in trials] #Ntrials x Nclust x samples					]
	th_mat = [t[1] for t in trials] #ntrials x n

	ntrials, nmodels, nsamples, ndens = len(trials), len(models), len(cd), 500
	
	upper = 7
	lower = 0
	x = np.linspace(lower, upper, ndens)	
	binsize = (upper-lower)/ndens

	modeldens = np.zeros([ntrials, nmodels, nsamples, ndens])	

	for trial_i, (trial_ws, th) in enumerate(zip(rlist, th_mat)):

		#empirical density
		#th = th[~np.isnan(th)]
		#density = gaussian_kde(th)		
		#totaldens = np.zeros([ndens])        
			
		for c, (m, ws) in enumerate(zip(models, trial_ws)):
			
			preds = m.predict(cd) #for each model predict the y value.  
			
			for t, (p, w) in enumerate(zip(preds, ws)): #for each t get the scaled density distribution
				dns = norm.pdf(x, p, m.spread) #evaluate prob density for empirical th range				
				#print('binsize', binsize)
				#print('weight', w)
				#print('1:', np.sum(dns))
				dns *= binsize #sum to one.
				#dns /= np.sum(dns)
				#plt.plot(dns)
				#plt.show()
				#print('2:', np.sum(dns))
				
				
				#dns /= np.sum(dns)
				dns *= w #the wights are given for each cluster or model
				#so for each time slice you have a multimodal distribution that sums to one.		
				#dns / nsamples
				#print(np.sum(dns))				
				modeldens[trial_i, c, t, :] = dns #/ nsamples	
												
				
			meandens = np.nansum(modeldens[trial_i, c, :, :], axis = 0)							
						
			#add to total dens
		#	totaldens += meandens		
		
	return(modeldens)

def add_vlines(ax, means):

	for i, mean in enumerate(means):
		ax.axvline(mean, linestyle = '--', color = ch.cluster_cols[i], alpha = .2)

@memoize
def central_tendancies(fit,dens_mat):

	"""dens_mat has shape (ntrials, nmodels, nsamples, ndens). Each trial matrix sums to one """

	ntrials, nmodels, nsamples, ndens = dens_mat.shape
	th = fit['data'][:,1]
	x = np.linspace(0, 7, ndens)	
	trial_clust_dens = np.array([np.nansum(m, axis = 0) for t in dens_mat for m in t]).reshape(ntrials, nmodels, ndens)
	mean_clust = [np.nanmean(trial_clust_dens[:, m, :],axis = 0) for m in range(nmodels)]

	modes = [np.argmax(m) for m in mean_clust]
	means = [np.dot(x, (m / np.sum(m))) for m in mean_clust]
	variances = [np.dot((x-mean)**2, (m / np.sum(m))) for m, mean in zip(mean_clust, means)]
	#means = [np.mean(np.random.choice(x, 10000, replace=True, p= (m / np.sum(m)))) for m in mean_clust]
	weights = np.array([np.sum(m) for m in mean_clust])
	weights /= np.sum(weights) #make into percentage.

	"""
	if plot:
		ax = plt.gcf().axes
		for i, (mode, mean, model) in enumerate(zip(modes, means, mean_clust)):
			ax[1].plot(x[mode], model[mode], 'o', color = ch.cluster_cols[i], alpha = .2)
			ax[1].axvline(mean, linestyle = '--', color = ch.cluster_cols[i], alpha = .2)
	"""
	
	return(x[modes], means, weights, variances)
	

def plot_state(ax, fit, fitting = False):

	data, models, resps, lik_hist = ch.dictget(fit, 'data', 'models', 'resps','lik_hist')

	
	margin = [.5,.5]            
	#for a in ax.flat: a.clear()
	cmap = np.array(ch.weighted_RGB(resps, ch.hex_to_RGB(ch.cluster_cols), norm = False))                    
	zord = {0: 3, 1: 2, 2: -1}					
	ax[0].scatter(data[:,0], data[:,1], alpha = .025, c = cmap)            
	for mi, m in enumerate(models):
		ax[0].plot(data[:,0], m.predict(data[:,0]), '-', color = ch.cluster_cols[mi], zorder = zord[mi])                            
		ax[1].plot(data[:,0], m.predict(data[:,0]), '-', color = ch.cluster_cols[mi], zorder = zord[mi])                            
		cis = norm.interval(.95, loc = m.predict(data[:,0]), scale = m.spread)        
		for ci in cis:            
			ax[1].plot(data[:,0], ci, '--', color = ch.cluster_cols[mi], alpha = .5, zorder = zord[mi])                            

	ax[0] = ch.limit_to_data(ax[0], data, margin)	
	ax[0].set_ylim(ymin = 0)
	ax[1] = ch.limit_to_data(ax[1], data, margin)
	ax[1].set_ylim(ymin = 0)

	
	"""
	if fitting:
		ax[2].plot(range(len(lik_hist)), lik_hist, color = ch.raw_col) 
	else:
		vangle = fit['vangle']
		ax[2].scatter(data[:,0], vangle, alpha = .25, c = cmap) 

		#hangle = fit['hangle']	
		#ax[3].scatter(data[:,0], hangle, alpha = .25, c = cmap) 
		yawrate = fit['yawrate']	
		ax[3].scatter(data[:,0], yawrate, alpha = .25, c = cmap) 

	"""
	return (ax)

def initial_values(y, yint = 10, spread = [.5,.5], noise_params = [2,5], clusters = ['gf', 'entry','noise']):
	
	mn = np.mean(y)
	var = np.var(y)
	n_clust = len(clusters)
	#if len(yint) == 0: yint = np.array([np.random.randint(5,25, n_clust)])
	props = np.full(n_clust, 1/n_clust) #this has an array of size n_clust summing to one, equal proportions.
	if 'noise' in clusters:
		props[-1] = .1
		props[:-1] = (1 - props[-1]) / (n_clust-1)                

	b0 =  {'gf': mn, 'entry':yint, 'exit':yint+10, 'noise':noise_params[0]}
	b1 = {'gf': 0, 'entry':-1, 'exit':-1, 'noise':0}    
	sds =  {'gf': spread[0], 'entry':spread[1], 'exit':spread[1], 'noise':noise_params[1]}    

	#this bit only necessary for cluster flexibility    
	b0s = []
	b1s = []
	spreads = []
	for c in clusters:
		b0s.append(b0[c])
		b1s.append(b1[c])    
		spreads.append(sds[c])
	return(n_clust, b0s, b1s, props, spreads)

class regmodel():

	def __init__(self, X, y, b0, b1, prop, spread, mtype, name):

		self.name = name
		self.mtype = mtype
		self.b0 = b0
		self.b1 = b1
		self.prop = prop        
		self.spread = spread
		self.residuals = (self.predict(X) - y)        
		self.residuals_w = (self.predict(X) - y)        
	
	def predict(self, X):
		
		return (np.dot(X, self.b1) + self.b0)        
		

	def fit(self, X, y, resp):
								
		self.b0 = np.average(y - np.dot(X,self.b1), weights = resp) #estimated intercept, given resps

		#update params
		self.residuals = (self.predict(X) - y)  #is it because these residuals are not weighted?		
		mean = np.average(self.residuals, weights = resp)  #weighted average, should be zero		
		var = np.average((self.residuals - mean)**2, weights = resp) #weighted variance
		self.spread = np.sqrt(var)                
		
	def loglik(self):        
		return norm.logpdf(self.residuals, 0.0, self.spread) + np.log(self.prop)

def loop_over_spread():
	pass


def publication_plot(fit, dens_mat, means, variances, weights):

	fig_cm = np.array([17,12])
	fig_inc = fig_cm /2.54 
	fig = plt.figure(constrained_layout = True, figsize = fig_inc)
	gs = fig.add_gridspec(6,3, width_ratios=[1,1,1.5])
	axes = [fig.add_subplot(gs[0:2, :-1]),
			fig.add_subplot(gs[2:4, :-1]),
			fig.add_subplot(gs[4:, :-1]),			
			fig.add_subplot(gs[:-3, -1]),
			fig.add_subplot(gs[3:, -1])]
	
	axes[0], axes[1] = plot_state([axes[0], axes[1]], fit)
	
	trials = map_trials(fit)		
	axes[2] = weights_over_time(trials, smooth = True, plot = True, ax = axes[2])
	

	axes[-2], axes[-1] = plot_average_dens(fit, dens_mat, means, variances, weights, ax = [axes[-2], axes[-1]])


	#labels and formatting

	#th vs veh time plots
	for a in [axes[0],axes[1]]:
		a.set_xlabel("Time into Trial (s)", fontsize = 10)
		a.set_ylabel("TH (s) ", fontsize = 10)
		a.set_ylim(0, 7)

	legend_elements = [Line2D([0], [0], color = ch.cluster_cols[0], lw=2, label='GF'),
						Line2D([0], [0], color = ch.cluster_cols[1], lw=2, label='EF'),
						Line2D([0], [0], color = ch.cluster_cols[2], lw=2, label='Noise')]
	axes[1].legend(handles=legend_elements, loc = 1, fontsize= 8, frameon = False)

	#weight plot
	axes[2].set_xlim(axes[0].get_xlim())
	axes[2].set_xlabel("Time into Trial (s)", fontsize = 10)
	axes[2].set_ylabel("Cluster Weight", fontsize = 10)
	axes[2].set_yticks([0,.5,1])  
	axes[2].set_yticklabels(['0','.5','1'])  

	#dens plots	
	for a in [axes[-2], axes[-1]]:
		a.set_yticks([	])  
		a.set_yticklabels([''])  
		a.set_xticks([0,1,2,3,4,5,6,7])  
		a.set_xticklabels(['0','1','2','3','4','5','6','7'])  
		a.set_xlabel("TH (s)", fontsize = 10)
		a.set_ylabel("Density", fontsize = 10)

	for a, label in zip(axes, ['A','B','C','D','E']):
		a.text(0.05, 0.95, label, transform=a.transAxes,
	  		fontsize=12, fontweight='bold', va='top')

	legend_elements = [Line2D([0], [0], color = 'xkcd:black', lw=1, label='Empirical'),
						Line2D([0], [0], color = 'xkcd:black', lw=1, ls='dotted', label='Fitted')]
	
	axes[-2].legend(handles=legend_elements, loc = 1, fontsize = 8, frameon = False)

	id_adj = int(fit['ID']) - 500
	dms = {0: 'Manual', 1: 'Auto-Replay', 2: 'Auto-Stock'}
	mytitle = 'Participant: {}, Driving Mode: {}'.format(str(id_adj), dms[fit['drivingmode']])
	fig.suptitle(mytitle, fontsize=12)	
	plt.savefig('Fits/linmix_fit_' + str(id_adj) + '_' + dms[fit['drivingmode']]	 + '.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
#	plt.savefig('Fits/linmix_sample_' + str(fit['ID']) +'_' + str(fit['drivingmode']) +'.svg', format='svg', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
	
	#plt.show()

def plot_results(fit, dens_mat, means, variances, weights):

	f1, ax = plt.subplots(4,1, figsize = (10,10), sharex = True, sharey= False)
	plot_state(ax, fit)                   

	trials = map_trials(fit)							
				
	f2 = plt.figure("weights", figsize=(6,4))
	w_t = weights_over_time(trials, smooth = False, plot = True)
	ch.move_figure(f2, .52,.99)
				
	#print("here")
	#f3 = plt.figure("th density on smoothed weights", figsize=(6,4))
	#plot_density(fit, trials, w_t)
	#ch.move_figure(f3, .52,.49)

	#dens_mat = plot_density_trials(fit, trials, plot = False)
	f4 = plt.figure("th density on fitted weights", figsize=(6,4))
	plot_average_dens(fit, dens_mat, means, variances, weights)

	ch.move_figure(f4, .52,.49)

	#add_vlines(f4.axes[1], means)
	plt.draw()
	plt.pause(.01)
	input('enter to close')
	plt.close('all')

def save_average_pp_resps(fits):
	print('mapping weights')
	master = pd.DataFrame()
	for fit in fits:
		
		print(fit['ID'])
		print('mapping trials')
		trials = map_trials(fit)		
		print(len(trials))		
		rlist = np.array([t[2][0] for t in trials]).T #rlist = Ntrials x Nsamples of gf weights.
		print(rlist.shape)
		avg_r = np.nanmean(rlist, axis =1)
		print(avg_r)		
		trial = pd.DataFrame()
		trial['w'] = avg_r
		trial['ID'] = fit['ID']
		trial['drivingmode'] = fit['drivingmode']
		
		master = pd.concat([master, trial])

	return(master)

def summarise_fits(fits, plot = True):

	columns = ['clust_n','intercept','slope','spread','name','modeltype','mode','mean','weight','variances','drivingmode','ID', 'dataset','entropy']
	output = []
	master = []
	for fit in fits:


		print(fit['ID'])
		print('mapping trials')
		trials = map_trials(fit)
		
		print('estimating density')
		
		dens_mat = estimate_density_matrix(fit, trials)
		print('calculating central tendancies')
		modes, means, weights, variances = central_tendancies(fit, dens_mat)

		entropy = entropy_scores(fit)		

		if plot:
			#plot_results(fit, dens_mat, means, variances, weights)			
			publication_plot(fit, dens_mat, means, variances, weights)			
												
		for i, (md, mn, wt, var, mod) in enumerate(zip(modes, means, weights, variances, fit['models'])):
				out = [i, mod.b0, mod.b1, mod.spread, mod.name, mod.mtype, md, mn, wt, var, fit['drivingmode'], fit['ID'], fit['dataset'], entropy]
				output.append(out)

	res = pd.DataFrame(output, columns = columns)	
	
	return(res)

def loop_trials_fit(df, yint, plot = True):
   
	fits = []
	for g, d in df.groupby(['trialcode']): 		
		
		vangle = d.vangle.values
		hangle = d.hangle.values
		yawrate = d.yawrate.values
		cumdist = d.midline_cumdist.values / 8.0   
		th_actual = d.th_along_midline.values
		
		X = np.transpose(np.array([cumdist, th_actual]))
		X = X[~np.isnan(X).any(axis=1)].reshape(-1,2)
								
		print(X.shape[0])
		print(g)
		
		init_vals  = initial_values(X[:,1], yint, spread = .5)

		try:

			fit = mixlin_em(X, *init_vals, anim = False, verbose = False)     
			fit['drivingmode'] = d.drivingmode.values[0]
			fit['ID'] = d.ID.values[0]
			fit['trialcode'] = d.trialcode.values[0]
			fit['dataset'] = d.dataset.values[0]

			"""
			best_fit = []
			best_spread = []
			lik = -np.inf
			for s in np.arange(.1,2,.1):
				fit = mixlin_em(X, n_clust, ints, slo, props, s, anim = False, verbose = False)     
				ll = fit['lik_hist'][-1]
				print('fit loglik = ', ll)
				if ll < lik: best_fit, best_spread == fit, s
			"""		
			fit['vangle'] = vangle
			fit['hangle'] = hangle
			fit['yawrate'] = yawrate
			fits.append(fit)
			if plot:
				summarise_fits([fit], plot)
		except AssertionError as e:
			print(e)
			try: 
				fit = mixlin_em(X, *init_vals, anim = True, verbose = True)     				
			except:
				plt.close('all')
				continue
		except Exception as e:
			print(e)
			break
				#columns = ['clust_n','mode','mean','weight','drivingmode','ID']
	
	return (fits)

def plot_frames(df):

	fig, ax = plt.subplots()
	for i, row in df.iterrows():
		viewpoint = row.posx, row.posz
		yaw = row.yaw
		ax = perspective(viewpoint, yaw, ax)
		plt.draw()
		ax.plot(row.hangle, row.vangle, 'bo')
		ax.plot(row.midline_ref_onscreen_x, row.midline_ref_onscreen_z, 'go')
		#title = ' '.join([str(row.ID), str(row.th_along_midline)])
		title = ','.join([str(round(row.midline_ref_world_x,2)), str(round(row.midline_ref_world_z,2)), str(round(row.th_along_midline,2))])
		plt.title(title)
		plt.pause(.1)

def grid_initial_values(X, yint, noise_params):

	"""gf should always centre on the mean. Only grid Entry intercept and GF & Entry spreads"""

	gf_s = np.linspace(.25, 2, num = 3)
	e_s = np.linspace(.25, 2, num = 3)
	yints = np.linspace(9, 11, num = 3)

	Y, E, G = np.meshgrid(yints, e_s, gf_s)
	inits = np.array([Y.ravel(), E.ravel(), G.ravel()]).T
	init_list = []		
	for row in inits:				
		init_vals  = initial_values(X[:,1], row[0], spread = [row[1],row[2]], noise_params = noise_params)
		init_list.append(init_vals)

	return(init_list)
#@memoize
def loop_pp_fit(df, yint, plot = True):

	"""loop over participants"""
	#TODO: check angle distance to midline

	#exclude datapoints that are very far away from the midline.
	exclude_crit = 20 #20 degrees
	dists = np.sqrt(df.midline_vangle_dist.values ** 2 + df.midline_hangle_dist.values ** 2)			
	df = df.iloc[dists<exclude_crit,:]

	#exclude looking at markers
	df = df.iloc[(df.screen_x_pix.values> 260) & (df.screen_x_pix.values<1670), :]
	df = df.iloc[(df.screen_y_pix.values< 590), :]
	noise_spread = np.sqrt(np.var(df.th_along_midline.values))*2
	noise_mean = np.mean(df.th_along_midline.values)
	fits = []
	drivingmodes = [0,1,2]
	for drive in drivingmodes:
		query_string = "drivingmode == {}".format(drive)  
		print(query_string)
		data = df.query(query_string).copy()						

		for g, d in data.groupby(['ID']): 
			
			print(g, drive)
			ppid = g
			
			d.sort_values(['block','currtime'], inplace=True)	
			
			vangle = d.vangle.values
			hangle = d.hangle.values
			yawrate = d.yawrate.values
			cumdist = d.midline_cumdist.values / 8.0   
			th_actual = d.th_along_midline.values			

			#for debugging, check the capping at 10s

			"""
			excl = th_actual < 4
			plot_frames(d.iloc[excl,:])
			continue

			"""

			#plt.plot(dists, vangle)
			#plt.show()
			
			X = np.transpose(np.array([cumdist, th_actual]))
			#X = X[include, :] #only keep within midline envelope
			X = X[~np.isnan(X).any(axis=1)].reshape(-1,2)
			
			
			#loop over initial values						
			init_list = grid_initial_values(X, yint, [noise_mean, noise_spread])			
			#print(init_list)

			
			best_fit, best_inits = [],[]
			best_lik = -np.inf
			worst_lik = np.inf
			tots = len(init_list)
			print(f"beginning gridding...{tots} fits")
			for i, init_vals in enumerate(init_list):
				#print(f"fitting {i+1} / {tots}")
				fit = mixlin_em(X, *init_vals, anim = False, verbose = False)     
				if fit == []: 
					print("empty fit")
					continue
				ll = fit['lik_hist'][-1]
				#print(init_vals)
				#print('fit loglik = ', ll)
				if ll < worst_lik: worst_lik = ll
				if ll > best_lik: 					
					best_lik = ll
					best_fit, best_inits = fit, init_vals
						
			print("worst_lik: ", worst_lik)			
			print("best_lik: ", best_lik)			
			print("change %: ",  (best_lik - worst_lik)/worst_lik)
			#print (fit)			
			
			fit['drivingmode'] = drive
			fit['ID'] = ppid
			fit['dataset'] = d.dataset.values[0]
			#fit['block'] = block
			"""
			best_fit = []
			best_spread = []
			lik = -np.inf
			for s in np.arange(.1,2,.1):
				fit = mixlin_em(X, n_clust, ints, slo, props, s, anim = False, verbose = False)     
				ll = fit['lik_hist'][-1]
				print('fit loglik = ', ll)
				if ll < lik: best_fit, best_spread == fit, s
			"""		
			fit['vangle'] = vangle
			fit['hangle'] = hangle
			fit['yawrate'] = yawrate
			fits.append(fit)
			if plot:
				summarise_fits([fit], plot)
			
			"""
			except AssertionError as e:
				print(e)
				continue
			except Exception as e:
				print(e)
				break
			"""
				#columns = ['clust_n','mode','mean','weight','drivingmode','ID']
	
	return (fits)

def mixlin_em(data, n_clust, b0, b1, props, spreads, clust_types = ['gf','entry','noise'], niter = 300, tol = 1e-4, anim = False, verbose = True):

	y = data[:,1]
	X = data[:,0]
	model_types = {'gf':'empty','entry':'slope','exit':'slope','noise':'empty'}
	n_clust = len(clust_types)
	#beta = [b0, b1] #intercept, slope.

	#initialise models.
	
	models = []
	for ci, c in enumerate(clust_types):
		mtype = model_types[c]   
		model = regmodel(X, y, b0[ci], b1[ci], props[ci],spreads[ci], mtype, name = c)
		models.append(model)

	lik_hist = [] #for convergence
	n_obs, n_var = data.shape
	liks = np.zeros([n_clust,n_obs])
	resps = softmax(liks, axis = 0)        

	if anim:
		fig, ax = plt.subplots(3,1, figsize=(10,10))        
	
	for i in range(niter):
		#print(i)

		if anim:
			fit = {
					'data':data,
					'models':models,                    
					'resps':resps,
					'lik_hist':lik_hist,                    
				}         
			ax = plot_state(ax, fit, fitting = True)
			plt.draw() 
			if i == 0: plt.pause(2)
			plt.pause(.5)

		if verbose: 
			for m in models:
						
				print("name: {}\tmtype: {}\tb0: {}\tb1: {}\tw: {}\tspread: {}\tll: {}".format(m.name, m.mtype, m.b0, m.b1, m.prop, m.spread, logsumexp(m.loglik())) )

		#E step. compute weights.        
		for mi, m in enumerate(models):    
									
			#loglik = norm.logpdf(y, m.predictions, m.spread) + np.log(m.prop) #update to log
			loglik = m.loglik()                       
			liks[mi] =  loglik			
			   
		
		total_lik = np.sum(logsumexp(liks, axis=0))
		lik_hist.append(total_lik)        
		
		resps = softmax(liks, axis=0)
		cluster_size = np.sum(resps, axis = 1)
		
		props = cluster_size / n_obs        
		
		if 'noise' in clust_types:  #update noise cluster but make all other clusters equal.          
			shared_size = np.sum(props) - props[-1]            
			props[:-1] = shared_size / (n_clust-1)
		

		#M step
		if i >  0:                        
			lik_change =  total_lik - lik_hist[-2]
			if verbose:
				print('likchange:', lik_change)			
			assert(lik_change > 0)			
			if lik_change < tol:

				print("converged on step: ", i)   
				if anim:
					plt.close('all')
				
				fit = {
					'data':data,
					'models':models,                    
					'resps':resps,
					'lik_hist':lik_hist,                    
				}
				return (fit)

		#do weighted linear regression to obtain new fits.
		for m, resp, prop in zip(models, resps, props):
			if m.name != 'noise': m.fit(X, y, resp)                   
			m.prop = prop  

#w = np.array([.5,.5,0,0]).T.reshape(1,-1)
#print(weighted_RGB(w, hex_to_RGB(cols)))

def get_point_cumdist(gm, tree, cumdist):    
	
	_, closest = tree.query(gm)
	gm_cumdist = cumdist[closest]

	return(gm_cumdist)

def get_yintercept(x,slope=-1):
	"""x is point in cumdist line, slope is -1"""
	return (-slope*x)

#res = load_cluster_fits()
#preds, clusters = predict_trial(res.query('ID == 501 & drivingmode == 0'))
#regs = linfits(clusters)
def add_matched_trial_column(df):

	ID = df.ID.values
	block = df.block.values
	condition = df.condition.values
	count = df['count'].values

	match = ['_'.join([str(int(i)),str(int(b)),str(int(cd)),str(int(cn))]) for i,b,cd,cn in zip(ID, block, condition, count)]
	df['match'] = np.array(match)
	return (df)

def remove_unmatched(df):

	replay_df = df.query('drivingmode == 1')
	replay_matched = set(replay_df.match.values) #list of all replayed trials
	
	new_df = df.query('drivingmode == 2 | match in @replay_matched')
	
	return (new_df)

def main(plot = True, outfile = 'linmix_res.csv', datafile = '../Data/trout_subset_2.feather'):

	file = ch.check_exist(outfile)
	print(file)

	ml = ch.get_midline()
	tree = ch.get_tree(ml)
	cumdist = ch.get_cumdist(ml)    

	gazemodes = np.array([[-22.74,70.4],[25,50.39]])
	#gazemodes = np.array([[-22.74,70.4]])
	gm_cumdist = get_point_cumdist(gazemodes, tree, cumdist)
	gm_cumdist /= 8.0 #convert to time
	yints = get_yintercept(gm_cumdist)
	yints = yints[0]
	
	df = load_data(datafile)

	#remove manual trials that do not have a replay match
	df = add_matched_trial_column(df)
	df = remove_unmatched(df)	

	#df = df.query("ID == 504 & drivingmode == 1")
	fits = loop_pp_fit(df, yints, plot = plot)
	#fits = loop_trials_fit(df, yints, plot = plot)
	print('finished fitting')
	print('starting summarising...')
	#out = save_average_pp_resps(fits)
	#out.to_csv('ppweights.csv')

	res = summarise_fits(fits, plot = plot)

	out = pd.DataFrame(res)
	res.to_csv(file)                                    

if __name__ == '__main__':

	#run_simulated_gmm()
	main(plot=True, outfile = 'linmix_d1.csv', datafile = '../Data/trout_6.feather')