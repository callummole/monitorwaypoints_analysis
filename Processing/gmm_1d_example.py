import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.special import softmax, logsumexp

raw_col = '#029386'
cluster_cols = {0: '#0343df', 1: '#e50000', 2:'#f97306', 3: '#15b01a'}  


def simulate_data(plot = True):

	nsamples = 5000

	a_sd = 1
	a_mean = 3
	a_prop = 2

	b_mean = -2
	b_sd = 2
	b_prop = 1
	proportions = [a_prop, b_prop]
	proportions /= np.sum(proportions, axis = 0)
	#print(props)
	a = np.random.randn(int(nsamples*proportions[0]))*a_sd + a_mean
	b = np.random.randn(int(nsamples*proportions[1]))*b_sd + b_mean

	X = np.concatenate( (a,b) )

	
	if plot: 
		density = gaussian_kde(X)
		rg = np.linspace(min(X),max(X),500)
		plt.plot(rg,density(rg), raw_col, linewidth = 1) #row=0, col=0
		plt.plot(rg, norm.pdf(rg, a_mean, a_sd)*proportions[0], color = cluster_cols[0], linewidth = 1)        
		plt.plot(rg, norm.pdf(rg, b_mean, b_sd)*proportions[1], color = cluster_cols[1], linewidth = 1)
		plt.show()

	return(X)

def gmm_em(x, n_clust, plot = True, iter = 100, tol = 1e-4):

	"""fits a guassian mixture model through expectation-maximisation"""

	#initialise means and sds and props
	means = np.random.randn(n_clust)
	sds = np.ones(n_clust)
	props = np.full(n_clust, 1/n_clust) #this has an array of size n_clust summing to one
	n_obs = x.shape[0]
	lik_hist = [] #for convergence

	for i in range(iter):

		if plot:
			plt.cla()
			density = gaussian_kde(x)
			rg = np.linspace(min(x),max(x),500)
			plt.plot(rg, density(rg), color = raw_col)
			for c, (m, s, p) in enumerate(zip(means, sds, props)):
				plt.plot(rg, norm.pdf(rg, m, s)*p, color = cluster_cols[c])                			
			plt.draw()
			plt.pause(.2)

		#EXPECTATION STEP
		#compute likelihood of latent labels (i.e. belonging to each cluster)		
		liks = np.zeros([n_clust,n_obs])
		for c in range(n_clust):    
			prior = props[c] #props is cluster size
			loglik = norm.logpdf(x, means[c], sds[c]) + np.log(props[c])
			liks[c] =  loglik
			   			
		total_lik = np.sum(logsumexp(liks, axis=0))                
		lik_hist.append(total_lik)        

		resps = softmax(liks, axis=0) #responsibility vector for each sample		
		
		props = np.sum(resps, axis = 1)     #to update estimates of cluster size take sum of total weights and divide by n observations
		props /= n_obs			

		#check convergence
		if i >  0:                        
			lik_change =  total_lik - lik_hist[-2]
			# print(lik_change)
			assert(lik_change > 0)
			if lik_change < tol:         
				fit = {
					'data':X,
					'means':means,
					'sds':sds,
					'props':props,
					'lik_hist':lik_hist,
					'resps':resps                    
				}
				print("converged on step: ", i)   
				return (fit)  

		#MAXIMISATION STEP to obtain new means and sds, given the expectations/responsibilities
		means = [np.average(x, weights = r) for r in resps] #weighted average 
		sds = [ np.average((x-m)**2, weights = r) for m, r in zip(means, resps)] #weighted variance
		sds = np.sqrt(sds)

if __name__ == '__main__':
	X = simulate_data(plot = True)
	gmm_em(X, 2, plot = True)

