import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal, gaussian_kde, t, stats
from memoize import memoize


def load_data():
		
	df = pd.read_feather('../Data/trout_subset.feather')
	#df = pd.read_csv("../Data/trout_rerun_gazeandsteering_2019-11-21.csv")
	#query_string = "drivingmode < 3 & confidence > .8"	
	#query_string = "drivingmode < 3 & confidence > .6 & dataset == 1 & on_srf == True"
	query_string = "drivingmode < 2 & roadsection < 2 & T0_on_screen ==0 & dataset == 2"
	df = df.query(query_string).copy()
	#correct for typo and set nans to unwanted data
	df.loc[df.ID == 203, 'ID'] = 503  
	df.loc[df.confidence < .6, 'th_along_midline'] = np.nan
	df.loc[df.on_srf == False, 'th_along_midline'] = np.nan


	weights = pd.read_csv('ppweights.csv')

	return(df, weights)


def add_matched_trial_column(df):

	ID = df.ID.values
	block = df.block.values
	condition = df.condition.values
	count = df['count'].values

	match = ['_'.join([str(int(i)),str(int(b)),str(int(cd)),str(int(cn))]) for i,b,cd,cn in zip(ID, block, condition, count)]
	df['match'] = np.array(match)
	return (df)

def interp(x, y, z):
	"""x is original index, y is predicted value, z is new index to interpolate on"""
	interp = interp1d(x, y, bounds_error = False, fill_value = np.nan)
	return(interp(z))

def exist_array(x, ref):
		#select only close items of cd_ref to interp on
		x_rnd = np.round(x,2)
		cd_exist = []
		for cd in x_rnd:			
			if cd in ref:				
				cd_exist.append(cd)				
		return(np.array(cd_exist))

def pad_array(y, trial_ref, cd_ref):
	out = np.empty(len(cd_ref))
	trial_i = 0
	for i, cd in enumerate(cd_ref):
		if cd in trial_ref:
			yout = y[trial_i]
			trial_i += 1
		else:
			yout = np.nan
		out[i] = yout		
	return(out)

def interpolate_weights(df, cd_ref):

#	pps = list(set(df.ID.values))#
	#print(pps)
	PP = []
	cd_map = np.arange(0, 20, step = .01).reshape(-1,1) #constant array throughout all trials
	cd_map = cd_map * 8

	for p,d in df.groupby(['ID']):		
		
		wts =d.w.values.reshape(3, -1).T #3 driving modes				
		wts = np.hstack([wts, cd_map])		
		sh = wts.shape		
		print(sh)
		wts = wts[~np.isnan(wts).any(axis=1)].reshape(-1, sh[1])		
		mn_wt = np.nanmean(wts[:,:-1], axis = 1)		
		interp_wts = interp(wts[:,-1], mn_wt, cd_ref)				
		PP.append(interp_wts)
	
	return(PP)

@memoize
def interpolate_th(df, cd_ref):

	"""map all trials onto cd_ref"""

	#empty list for each ID.
	pps = list(set(df.ID.values))
	print(pps)

	PP = [[] for p in pps] 

	replay_df = df.query('drivingmode == 1')
	matched = set(replay_df.match.values) #list of all replayed trials

	for code in matched:
			
		matchdata = df.query(f"match == '{code}'")
		
		match_signals = np.empty([len(cd_ref), 2])
		data_size = np.empty([2])
		ppid = matchdata.ID.values[0]		
		ppidx = pps.index(ppid)		

		col = {0: 'b', 1: 'r'}
		for g, d in matchdata.groupby(['drivingmode']):        
			
			#interp to cd_ref
			x = d.midline_cumdist.values
			y = d.th_along_midline.values            
			#print(x)
			#print(y)
						
			#trial_ref = exist_array(x, cd_ref)			
			interp_th = interp(x, y, cd_ref)	
			#th = pad_array(interp_th, trial_ref, cd_ref)
			
			match_signals[:,int(g)] = interp_th
			#data_size[int(g)] = len(y)
			#plt.plot(x, y, '.-', color = col[int(g)], alpha = .5)

		"""		
		plt.plot(cd_ref, match_signals[:,0],'b.')    
		plt.plot(cd_ref, match_signals[:,1],'r.')    
		plt.title(code)
		plt.show()

		"""
		PP[ppidx].append([code,match_signals])


	return (PP)

@memoize
def pp_subtraction(match_trials, cd_ref):
	"""for each participant create a difference signal"""	

	PP = np.empty([len(cd_ref), len(match_trials)])
	
	for pi, pp_matches in enumerate(match_trials):
		print('trials:', len(pp_matches))
		D = np.empty([len(cd_ref), len(pp_matches)])
		for mi, match in enumerate(pp_matches):			
			ths = match[1]
			diffs = np.empty(ths.shape[0])
			for i, cd in enumerate(ths):
				if np.any(np.isnan(cd)): 
					diff = np.nan
				else:
					diff = cd[1] - cd[0] #passive - active
				diffs[i] = diff
			D[:,mi] = diffs
		
		avg = np.nanmean(D, axis = 1)
		
		PP[:, pi] = avg
	med = np.nanmedian(PP, axis = 1)

	return(med, PP)

def CIs(d, ci = .99):

    m = np.nanmean(d)
    n = len(d)
    sd = np.sqrt (np.sum( (d - m)**2 ) / (n-1))
    se = sd / np.sqrt(n)
    cis = se * t.ppf( (1+ci) / 2, n-1)  
      
    return(m, cis)

if __name__ == '__main__':

	print('loading data')
	df, weights = load_data()

	"""
	for pp, d in df.groupby(['ID']):

		print(pp)
		blocks = list(set(d.block.values))
		print(blocks)
		print(len(blocks))
	
	"""
	print('adding column')
	df = add_matched_trial_column(df)

	#interpolate th_along_midline signal for each time series, based on midline_cumdist.
	cumdist = df.midline_cumdist.values
	cd_ref = np.arange(round(min(cumdist),2), round(max(cumdist),2), step = .01)	
	cd_ref = np.round(cd_ref,2) #ensure exact same as later rounding process 
	print('interpolating')
	weights = interpolate_weights(weights, cd_ref)	
	match_trials = interpolate_th(df, cd_ref)
	

	#use this to subtract the matched trials from one another.
	print('subtracting')
	overall_med, PPs = pp_subtraction(match_trials, cd_ref)

	#plt.plot(cd_ref, avg, '-', alpha = .1)		
	#plt.plot(cd_ref, overall_med, color = 'k', alpha = .8)		
	#plt.axhline(y=0, color = (.8,.8,.8), linestyle = '--')
	#plt.show()

	#overall_med = overall_med[~np.isnan(overall_med)]	
	avgs = []
	x = np.linspace(-2, 2, 500)
	col = np.array([0,0,1])
	

	print(PPs.T.shape)
	print("weight len:", len(weights))
	dens = []
	for pp, wt in zip(PPs.T, weights):		

		X = np.array([pp, wt]).T
		sh = X.shape
		X = X[~np.isnan(X).any(axis=1)].reshape(-1,sh[1])
		
		
		mn = np.average(X[:,0], weights = X[:,1])
		print("mean", mn)
		avgs.append(mn)		

		"""
		c = np.tile(col, X.shape[0]).reshape(X.shape[0],3)
		wt = X[:,1].reshape(-1,1) *.05		
		c = np.hstack([c, wt])
		y = X[:,0]
		plt.scatter(range(len(y)),y, c = c)
		"""
		

		density = gaussian_kde(X[:,0], weights = X[:,1])
		dens.append(density(x))

	#plt.show()
	
	print(stats.ttest_1samp(np.array(avgs),0))

	mn, ci = CIs(np.array(avgs), ci = .95)
	for ds in dens:
		plt.plot(x, ds, alpha = .4)
	plt.axvline(x = mn, linestyle = '--', color = 'b')
	plt.axvline(x = mn-ci, linestyle = '--', color = 'g')
	plt.axvline(x = mn+ci, linestyle = '--', color = 'g')
	plt.axvline(x = 0, linestyle = '--', color = (.8,.8,.8))
	plt.xlabel('gaze time headway')    
	plt.show()


	#plot the quantiles

	#

