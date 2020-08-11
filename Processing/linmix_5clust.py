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

"""
mixture linear regression model with four models.
mu = intercept
gf: th ~ N(mu0, sd0)
entry: th ~ N(mu1 + beta1*time, sd1)
obj1 : th ~ N(mu2 + beta2*time, sd2)
noise: th ~ N(mu3, sd3)

"""

def load_cluster_fits():

	res = pd.read_csv("gmm_res.csv")   
	return(res)

def load_data():

	#df = pd.read_feather('../Data/trout_2.feather')
	df = pd.read_feather('../Data/trout_subset.feather')
	#query_string = "drivingmode < 3 & confidence > .8"	
	#query_string = "drivingmode < 3 & confidence > .6 & dataset == 1 & on_srf == True"
	query_string = "drivingmode < 3 & confidence > .6 & on_srf == True"
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

def weights_over_time(trials, smooth = True, plot = True):

	"""returns smoothed coefficients per time point"""

	cd = trials[0][0]	
	rlist = np.array([t[2] for t in trials]) #Ntrials x Nclust x samples	
	if smooth: rlist = np.array([scipy.ndimage.gaussian_filter1d(r, 10) for rs in rlist for r in rs]).reshape(rlist.shape) #gauss filter with 10cm step
		
	w_t = [np.nanmean(rlist[:,c,:], axis = 0) for c in range(rlist.shape[1])] ##average across trials weight over time, w_t, for each cluster 			
	w_t /= np.sum(w_t, axis = 0) #sum to one.	

	if plot:
		for i, w in enumerate(w_t):
			plt.plot(cd, w, '.', color = ch.cluster_cols5[i], alpha = .2)

	return(w_t)

def map_trials(fit):

	"""for each block return a list of trials of equal length.

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
	
	cd = trials[0][0]
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
			plt.plot(x, dns, '-', color = ch.cluster_cols5[i])
			plt.title("At cumdist: " + str(cumdist[t]))
			plt.draw()
			plt.pause(.016)
			"""			   		
		meandens = np.nansum(modeldens[i, :, :], axis = 0)				

		ax[1].plot(x, meandens, '-', color = ch.cluster_cols5[i])
		#plt.plot(x, meandens, '-', color = ch.cluster_cols5[i])
				
		#add to total dens
		totaldens += meandens

	ax[0].plot(x, totaldens, 'xkcd:magenta')    
	
def plot_average_dens(fit, dens_mat):
	"""dens_mat has shape (ntrials, nmodels, nsamples, ndens). Each trial matrix sums to one"""

	ntrials, nmodels, nsamples, ndens = dens_mat.shape
	th = fit['data'][:,1]
	x = np.linspace(0, 7, 500)
	density = gaussian_kde(th)


	ax = plt.gcf().subplots(1,2)
	ax[0].plot(x,density(x), ch.raw_col) #row=0, col=0
	ax[1].plot(x,density(x), ch.raw_col) #row=0, col=0
	plt.xlabel('gaze time headway')    

	trial_clust_dens = np.array([np.nansum(m, axis = 0) for t in dens_mat for m in t]).reshape(ntrials, nmodels, ndens)
	mean_clust = [np.nanmean(trial_clust_dens[:, m, :],axis = 0) for m in range(nmodels)]
	
	for i, m in enumerate(mean_clust):
		ax[1].plot(x, m, color = ch.cluster_cols5[i])
	
	totaldens = np.sum(np.array(mean_clust), axis = 0)
	ax[0].plot(x, totaldens, 'xkcd:magenta')  	


def estimate_density_matrix(fit, trials, plot = False):
	
	models = fit['models']	
	th = fit['data'][:,1]
	
	cd = trials[0][0]
	rlist = [t[2] for t in trials] #Ntrials x Nclust x samples					]
	th_mat = [t[1] for t in trials] #ntrials x n

	ntrials, nmodels, nsamples, ndens = len(trials), len(models), len(cd), 500
	
	x = np.linspace(0, 7, ndens)	
	modeldens = np.zeros([ntrials, nmodels, nsamples, ndens])	

	for trial_i, (trial_ws, th) in enumerate(zip(rlist, th_mat)):

		#empirical density
		th = th[~np.isnan(th)]
		density = gaussian_kde(th)		
		totaldens = np.zeros([ndens])        
			
		for c, (m, ws) in enumerate(zip(models, trial_ws)):
			
			preds = m.predict(cd) #for each model predict the y value.  
			
			for t, (p, w) in enumerate(zip(preds, ws)): #for each t get the scaled density distribution
				dns = norm.pdf(x, p, m.spread) #evaluate prob density for empirical th range				
				dns *= w		
				modeldens[trial_i, c, t, :] = dns / nsamples     								
				
			meandens = np.nansum(modeldens[trial_i, c, :, :], axis = 0)							
						
			#add to total dens
			totaldens += meandens		
		
	return(modeldens)

def add_vlines(ax, means):

	for i, mean in enumerate(means):
		ax.axvline(mean, linestyle = '--', color = ch.cluster_cols5[i], alpha = .2)


def central_tendancies(fit,dens_mat):

	"""dens_mat has shape (ntrials, nmodels, nsamples, ndens). Each trial matrix sums to one """

	ntrials, nmodels, nsamples, ndens = dens_mat.shape
	th = fit['data'][:,1]
	x = np.linspace(0, 7, ndens)	
	trial_clust_dens = np.array([np.nansum(m, axis = 0) for t in dens_mat for m in t]).reshape(ntrials, nmodels, ndens)
	mean_clust = [np.nanmean(trial_clust_dens[:, m, :],axis = 0) for m in range(nmodels)]

	modes = [np.argmax(m) for m in mean_clust]
	means = [np.dot(x, (m / np.sum(m))) for m in mean_clust]
	#means = [np.mean(np.random.choice(x, 10000, replace=True, p= (m / np.sum(m)))) for m in mean_clust]
	weights = np.array([np.sum(m) for m in mean_clust])
	weights /= np.sum(weights) #make into percentage.

	"""
	if plot:
		ax = plt.gcf().axes
		for i, (mode, mean, model) in enumerate(zip(modes, means, mean_clust)):
			ax[1].plot(x[mode], model[mode], 'o', color = ch.cluster_cols5[i], alpha = .2)
			ax[1].axvline(mean, linestyle = '--', color = ch.cluster_cols5[i], alpha = .2)
	"""
	
	return(x[modes], means, weights)
	

def plot_state(ax, fit, fitting = False):

	data, models, resps, lik_hist = ch.dictget(fit, 'data', 'models', 'resps','lik_hist')

	
	margin = [1,1]            
	for a in ax.flat: a.clear()
	cmap = np.array(ch.weighted_RGB(resps, ch.hex_to_RGB(ch.cluster_cols5)))                    
					
	ax[0].scatter(data[:,0], data[:,1], alpha = .05, c = cmap)            
	for mi, m in enumerate(models):
		ax[0].plot(data[:,0], m.predict(data[:,0]), '-', color = ch.cluster_cols5[mi])                            
		ax[1].plot(data[:,0], m.predict(data[:,0]), '-', color = ch.cluster_cols5[mi])                            
		cis = norm.interval(.95, loc = m.predict(data[:,0]), scale = m.spread)        
		for ci in cis:            
			ax[1].plot(data[:,0], ci, '--', color = ch.cluster_cols5[mi], alpha = .5)                            

	ax[0] = ch.limit_to_data(ax[0], data, margin)	
	ax[1] = ch.limit_to_data(ax[1], data, margin)

	if fitting:
		ax[2].plot(range(len(lik_hist)), lik_hist, color = ch.raw_col) 
	else:
		vangle = fit['vangle']
		ax[2].scatter(data[:,0], vangle, alpha = .25, c = cmap) 

		#hangle = fit['hangle']	
		#ax[3].scatter(data[:,0], hangle, alpha = .25, c = cmap) 
		yawrate = fit['yawrate']	
		ax[3].scatter(data[:,0], yawrate, alpha = .25, c = cmap) 

	return (ax)

def initial_values(y, yint = [], spread = .5, clusters = ['gf', 'entry','obj1','obj2','obj3','end','noise']):
	
	mn = np.mean(y)
	var = np.var(y)
	n_clust = len(clusters)
	if len(yint) == 0: yint = np.array([np.random.randint(5,25, n_clust)])
	props = np.full(n_clust, 1/n_clust) #this has an array of size n_clust summing to one, equal proportions.
	if 'noise' in clusters:
		props[-1] = .1
		props[:-1] = (1 - props[-1]) / (n_clust-1)                

	b0 =  {'gf': mn, 'entry':yint[0], 'obj1':yint[1], 'obj2':yint[2],'obj3':yint[3],'end':10, 'noise':mn}
	b1 = {'gf': 0, 'entry':-1, 'obj1':-1, 'obj2':-1, 'obj3':-1,'end':0, 'noise':0}    
	obj_sd = .5
	sds =  {'gf': spread, 'entry':spread, 'obj1':obj_sd, 'obj2':obj_sd, 'obj3':obj_sd,'end':obj_sd, 'noise':np.sqrt(var)*5}    

	#this bit only necessary for cluster flexibility    
	b0s = []
	b1s = []
	spreads = []
	for c in clusters:
		b0s.append(b0[c])
		b1s.append(b1[c])    
		spreads.append(sds[c])
	return(n_clust, b0s, b1s, props, spreads)

def grid_initial_values(clusters = ['gf', 'entry','obj1']):
	pass

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

		#fixed intercept in fixed model.								
		if not self.mtype == 'fixed':self.b0 = np.average(y - np.dot(X,self.b1), weights = resp) #estimated intercept, given resps

		#update params
		self.residuals = (self.predict(X) - y)  #is it because these residuals are not weighted?		
		mean = np.average(self.residuals, weights = resp)  #weighted average, should be zero		
		var = np.average((self.residuals - mean)**2, weights = resp) #weighted variance
		self.spread = np.sqrt(var)                
		
	def loglik(self):        
		return norm.logpdf(self.residuals, 0.0, self.spread) + np.log(self.prop)

def loop_over_spread():
	pass

def plot_results(fit, dens_mat, means):

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
	plot_average_dens(fit, dens_mat)
	ch.move_figure(f4, .52,.49)

	#add_vlines(f4.axes[1], means)
	plt.draw()
	plt.pause(.01)
	input('enter to close')
	plt.close('all')


def summarise_fits(fits, plot = True):

	columns = ['clust_n','mode','mean','weight','drivingmode','ID', 'dataset']
	output = []
	for fit in fits:

		print(fit['ID'])
		print('mapping trials')
		trials = map_trials(fit)
		print('estimating density')
		dens_mat = estimate_density_matrix(fit, trials)
		print('calculating central tendancies')
		modes, means, weights = central_tendancies(fit, dens_mat)

		if plot:
			plot_results(fit, dens_mat, means)			
												
		for i, (md, mn, wt) in enumerate(zip(modes, means, weights)):
				out = [i, md, mn, wt, fit['drivingmode'], fit['ID'], fit['dataset']]
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

def loop_pp_fit(df, yint, plot = True):

	#TODO: check angle distance to midline
	fits = []
	drivingmodes = [0,1,2]
	for drive in drivingmodes:
		query_string = "drivingmode == {}".format(drive)  
		print(query_string)
		data = df.query(query_string).copy()						

		for g, d in data.groupby(['ID']): 
			
			print(g)
			
			d.sort_values(['block','currtime'], inplace=True)	
			vangle = d.vangle.values
			hangle = d.hangle.values
			yawrate = d.yawrate.values
			cumdist = d.midline_cumdist.values / 8.0   
			th_actual = d.th_along_midline.values

			#exclude datapoints that are very far away from the midline.
			exclude_crit = 20 #20 degrees
			dists = np.sqrt(d.midline_vangle_dist.values ** 2 + d.midline_hangle_dist.values ** 2)
			include = dists<exclude_crit
			exclude = dists >= exclude_crit			

			#plot_frames(d.iloc[exclude,:])

			#for debugging, check the capping at 10s

			#excl = th_actual > 8
			#plot_frames(d.iloc[excl,:])
			#continue



			#plt.plot(dists, vangle)
			#plt.show()
			
			X = np.transpose(np.array([cumdist, th_actual]))
			X = X[include, :] #only keep within midline envelope
			X = X[~np.isnan(X).any(axis=1)].reshape(-1,2)
									
			init_vals  = initial_values(X[:,1], yint, spread = .5)

			#try:
			fit = mixlin_em(X, *init_vals, anim = False, verbose = False)     
			print (fit)
			if fit == []: continue
			fit['drivingmode'] = drive
			fit['ID'] = g
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
			fit['vangle'] = vangle[include]
			fit['hangle'] = hangle[include]
			fit['yawrate'] = yawrate[include]
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

def mixlin_em(data, n_clust, b0, b1, props, spreads, clust_types = ['gf','entry','obj1','obj2','obj3','end','noise'], niter = 500, tol = 1e-4, anim = False, verbose = True):

	y = data[:,1]
	X = data[:,0]
	model_types = {'gf':'empty','entry':'slope','obj1':'fixed','obj2':'fixed','obj3':'fixed','end':'empty','noise':'empty'}
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
		donotfit = ['obj1','obj2','obj3','end','noise']
		for m, resp, prop in zip(models, resps, props):
			if m.name not in donotfit: m.fit(X, y, resp)                   
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

def main(plot = True, outfile = 'linmix_res.csv'):

	file = ch.check_exist(outfile)
	print(file)

	ml = ch.get_midline()
	tree = ch.get_tree(ml)
	cumdist = ch.get_cumdist(ml)    

	gazemodes = np.array([[-22.74,70.4], #entry
	[25,52], #1st object
	[25,44], #2nd object
	[25,36]]) #3rd object
	
	gm_cumdist = get_point_cumdist(gazemodes, tree, cumdist)
	gm_cumdist /= 8.0 #convert to time
	yints = get_yintercept(gm_cumdist)

	df = load_data()
	fits = loop_pp_fit(df, yints, plot = plot)
	#fits = loop_trials_fit(df, yints, plot = plot)
	print('finished fitting')
	print('starting summarising...')
	res = summarise_fits(fits, plot = plot)

	res.to_csv(file)                                    

if __name__ == '__main__':

	#run_simulated_gmm()
	main(plot=True, outfile = 'linmix_res_5clust.csv')