import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import calhelper as ch

nsamples = 10000

a_sd = 1
a_mean = 3
a_prop = 2

cluster_cols = {0: '#0343df', 1: '#e50000', 2:'#f97306', 3: '#15b01a'}  

b_mean = 5.5
b_sd = 2
b_prop = 1
proportions = [a_prop, b_prop]
proportions /= np.sum(proportions, axis = 0)
#print(props)
a = np.random.randn(int(nsamples*proportions[0]))*a_sd + a_mean
b = np.random.randn(int(nsamples*proportions[1]))*b_sd + b_mean

c = np.concatenate( (a,b) )

#empirical density
density = gaussian_kde(c)
rg = np.linspace(min(c),max(c),500)

mn = np.mean(c)
med = np.median(c)
mode = rg[np.argmax(density(rg))]

	
def remove_tr_axes(ax):

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
"""
ax = plt.subplot(111)
ax.plot(rg,density(rg), ch.raw_col, linewidth = 2) #row=0, col=0
ax.set_ylabel('Density', fontsize = 16)
ax.set_xlabel('Time (s)', fontsize = 16)
ax.set_xlim(0, max(c))
ax.axvline(mn, linestyle = '--', color = ch.cluster_cols[0], alpha = .8)
ax.axvline(med, linestyle = '--', color = ch.cluster_cols[1], alpha = .8)
ax.axvline(mode, linestyle = '--', color = ch.cluster_cols[2], alpha = .8)
remove_tr_axes(ax)
plt.savefig('sampledensity_withlines.png', format='png', dpi=1200, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
plt.show()
"""
purple = '#7e1e9c'

ax = plt.subplot(111)
ax.plot(rg,density(rg), ch.raw_col, linewidth = 1.5) #row=0, col=0
ax.set_ylabel('Density', fontsize = 16)
ax.set_xlabel('Time (s)', fontsize = 16)
ax.set_xlim(0, max(c))
ax.plot(rg, norm.pdf(rg, a_mean, a_sd)*proportions[0], color = purple, linewidth = 1.5)
#ax.axvline(a_mean, linestyle = '--', color = ch.cluster_cols[0], alpha = .8)
ax.plot(rg, norm.pdf(rg, b_mean, b_sd)*proportions[1], color = purple, linewidth = 1.5)
#ax.axvline(b_mean, linestyle = '--', color = ch.cluster_cols[1], alpha = .8)
#remove_tr_axes(ax)
plt.axis("off")  
plt.savefig('purple.png', format='png', dpi=1200, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
plt.show()


def gmm_em(x, n_clust, iter = 100):

    #initialise means and sds and props
    means = np.random.randn(n_clust)
    sds = np.ones(n_clust)
    props = np.full(n_clust, 1/n_clust) #this has an array of size n_clust summing to one

    for i in range(iter):

        #compute likelihood of latent labels    
        #each datapoint has a likelihood of belonging to each cluster.
        #E step
        liks = np.empty([n_clust,len(x)])
        for c in range(n_clust):    
            prior = props[c] #props is cluster size
            lik = norm.pdf(x, means[c], sds[c]) #lik is a vector of cluster weights for each observation
            liks[c] =  lik * prior
               
        #print(np.sum(liks,axis = 0))
        liks /= np.sum(liks, axis = 0)
        
        #list comprehension       
        #M step
        props = np.sum(liks, axis = 1)     #to update estimates of cluster size take sum of total weights and divide by n observations
        props /= len(x)

        means = [np.average(x, weights = l) for l in liks] #weighted average 
        sds = [ np.average((x-m)**2, weights = l) for m, l in zip(means, liks)] #weighted variance
        sds = np.sqrt(sds)

        #print(means)
        
        rg = np.linspace(min(x),max(x),100)
        
        for m, s in zip(means, sds):
            plt.plot(rg, norm.pdf(rg, m, s))
            
        plt.show()
        
gmm_em(c, 2)

#plt.hist(c, bins=50)
#plt.show()
