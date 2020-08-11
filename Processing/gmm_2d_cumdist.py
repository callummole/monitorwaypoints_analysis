import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal, pearsonr
from scipy.special import softmax, logsumexp
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import pandas as pd
from pprint import pprint
from scipy.stats import gaussian_kde
import calhelper as ch

cols = {'gf': 'xkcd:blue', 'entry': 'xkcd:red', 'exit':'xkcd:orange', 'noise': 'xkcd:green'}  

#2d gaussians
#heavily borrowed from http://ethen8181.github.io/machine-learning/clustering/GMM/GMM.html

#k means
#https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a

#paper on noise cluster: https://core.ac.uk/download/pdf/12167176.pdf

#blog on evaluation metrics https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4
#thread on evaluating cluster goodness of fit https://stats.stackexchange.com/questions/21807/evaluation-measures-of-goodness-or-validity-of-clustering-without-having-truth

#gmm lecture http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf

def simulate_data():
    nsamples = 1500

    #x axis = th_to_object, y xis = th_actual
    a_means = [7.0, 2.0]  #gfs
    b_means = [3.0, 3.0] #lafs
    c_means = [6.0, 4.0]
    a_cov = [[3, 0],
            [0, .2]]
        
    x_var, y_var = 1.2, .3
    xy_var = .3
    #xy_var = np.sqrt(x_var * y_var)
    b_cov = [[x_var, xy_var], 
            [xy_var, y_var]]

    print(corr(b_cov))
    b_cov = equal_variances(b_cov)
    b_cov = constrain_corr(b_cov, rlower=.5, rupper=.5)
    #print(corr(cov))
    #print(cov)

    #var = [.6, .8]
    #covar = [.2, .2]
    #cov = [[var[0], covar[0]],
            #[covar[1],var[1]]]
    #cov = list(zip(var, covar))
    #cov = np.array([var, covar]).T
    #print(np.diag(np.flipud(b_cov)))

    """
    #test
    x = np.array([np.random.multivariate_normal(b_means, b_cov)
    for i in range(nsamples*10)])
    print(x.shape)    
    print(pearsonr(x[:,0], x[:,1]))
    """

    
    c_cov = [[5, 0],
            [0, 5]]

    true_means = [a_means, b_means, c_means]
    true_cov = [a_cov, b_cov, c_cov]
    #print(true_cov.shape)
    proportions = [.7, .2, .05]
    proportions /= np.sum(proportions, axis = 0)

    n_features = len(proportions)
    data = np.empty([nsamples, 2])
    cols = {0: 'r', 1: 'b', 2: 'g'}
    markers = {0: 'o', 1: 's', 2: '*'}
    for i in range(nsamples):
        # pick a cluster id and create data from this cluster    
        k = np.random.choice(n_features, size = 1, p = proportions)[0]    
        x = np.random.multivariate_normal(true_means[k], true_cov[k])
        data[i] = x   
        plt.plot(x[0], x[1], marker = markers[k], alpha = .5, color = cols[k])

    abline(1,0)
    plt.axis('equal')
    ##plt.show()

    return(data)

def corr(cov):

    var = np.diag(cov)
    covar = np.diag(np.flipud(cov))
    pearson = covar[0] / np.sqrt(var[0] * var[1])
    return(pearson)

#could 

def equal_variances(cov):

    var = np.mean(np.diag(cov))
    covar = np.diag(np.flipud(cov))
    new_cov = [[var, covar[0]],
            [covar[1],var]]

    return(new_cov)

def constrain_var(cov, margin, cap):
    #cap variance along axis
    var = np.array(np.diag(cov))
    var[margin] = np.clip(var[margin], 0, cap)
    return (np.diag(var))


def constrain_corr(cov, rlower = .5, rupper=1):
    #constrain covariance based on correlation range

    var = np.diag(cov)
    sd_prod = np.sqrt(var[0]*var[1])
    upper = sd_prod * rupper #set covar to this if you want corr == 1
    lower = sd_prod * rlower

    covar = np.diag(np.flipud(cov))
    covar = np.clip(covar, lower, upper)

    new_cov = [[var[0], covar[0]],
            [covar[1],var[1]]]

    return(new_cov)


def gmm_em(data, means, covs, props, n_clust, clust_types, niter = 600, tol = 1e-4, tol_steps = 0, noise = True, plot = True):

    #initialise means and sds and props
    cov_type = {'gf': 'diag', 'entry':'negative', 'exit':'negative', 'noise':'noise'}

    print("clusters:", n_clust)
    if noise: #constrain noise to mean and variance of data
        n_clust += 1
        clust_types.append('noise')
        noise_means = np.mean(data, axis = 0)         
        means.append(noise_means)
        noise_cov = np.diag(np.var(data, axis = 0)*5)
        covs.append(noise_cov)
        props = np.full(n_clust, 1/n_clust) #re-estimate proportions
        #print(noise_means)
        #print(noise_cov)

    lik_hist = [] #for convergence
    n_obs, n_var = data.shape
    liks = np.zeros([n_clust,n_obs])

    for i in range(niter):


        if plot:
            plt.plot(data[:,0], data[:,1], 'o', alpha = .05)
            plot_ellipses(means, covs, clust_types)
            plt.xlim(0, 25)
            plt.ylim(-5,10)
            plt.show()
        #compute likelihood of latent labels    
        #each datapoint has a likelihood of belonging to each cluster.
        #E step
        for c in range(n_clust):    
            prior = props[c] #props is cluster size
            cov = covs[c]
            mean = means[c]
           # print(clust_types[c])
           # print(prior, cov, mean)
            #lik = multivariate_normal.pdf(data, means[c], cov) #lik is a vector of cluster weights for each observation (responsibility vector)            
            #liks[c] =  lik * prior
            #could add priors to the proportions sizes.
            loglik = multivariate_normal.logpdf(data, mean, cov) + np.log(prior)
            #print('logpdf', multivariate_normal.logpdf(data, mean, cov))
            #print('logprior', np.log(prior))
            #print('loglik', loglik)
            liks[c] = loglik
                                                             
        #convergence check before M-step 
        total_lik = np.sum(logsumexp(liks, axis=0))
        lik_hist.append(total_lik)
        #loglik = np.sum(np.log(liks))
        
        #M step
        resps = softmax(liks, axis=0)
        
        cluster_size = np.sum(resps, axis = 1)
        props = cluster_size / n_obs

        if i >  0:                        
            lik_change =  total_lik - lik_hist[-2]
            # print(lik_change)
            assert(lik_change > 0)
            if lik_change < tol:         
                fit = {
                    'data':data,
                    'means':means,
                    'covs':covs,
                    'props':props,
                    'lik_hist':lik_hist,
                    'resps':resps,
                    'clust_types':clust_types
                }
                print("converged on step: ", i)   
                return (fit)  
        
        """
        if plot:
            fit = {
                    'data':data,
                    'means':means,
                    'covs':covs,
                    'props':props,
                    'lik_hist':lik_hist,
                    'resps':resps,
                    'clust_types':clust_types
                }
            plot_clusters(fit, title = None)
            plt.show()
        """

        #weighted covariance
        for c, ct in enumerate(clust_types):
            # means            
            m = np.dot(resps[c], data)            
            m /= cluster_size[c]
            means[c] = m

            #TODO: not sure if the current method is the MLE for constrained covariances.
            #weighted variance
            """
            if cov_type[c] == 'positive':

                weighted_data = data * resps[c,:]
                wm = np.mean(weighted_data.ravel())
            """

            #covar
            diff = (data - m).T      
            wsum = np.dot(resps[c,:] * diff, diff.T)
            cov = wsum / cluster_size[c]
            #if cov_type[c] == 'diag': cov = constrain_var(np.diag(np.diag(cov)), margin =1, cap = .3) #re-create a diag from just the vairances
            if cov_type[ct] == 'diag': cov = np.diag(np.diag(cov)) #re-create a diag from just the vairances
            if cov_type[ct] == 'negative': cov = constrain_corr(equal_variances(cov), rlower = -1, rupper = 0) #constrain to at least .5
            if cov_type[ct] == 'noise': means[c], cov = noise_means, noise_cov #keep noise cluster contrained to means and variances                

            covs[c] = cov


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    ylim = axes.get_ylim()
   # print(ylim)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'k--')
    plt.ylim(ylim[0], ylim[1] + 1)
    

def plot_ellipses(means, covs, clust_types):
    
    for i, (m, c, ct) in enumerate(zip(means, covs, clust_types)):
        print("corr:", corr(c))
        ax = plt.gca()
        U, s, Vt = np.linalg.svd(c)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)        
        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(m, nsig * width, nsig * height,
                            angle, alpha = .2, color = cols[ct]))
    #plt.ylim(0,12)
    #plt.xlim(0,12)
                        

def plot_clusters(fit, title = None):


    data, means, covs, props, resps, clust_types = ch.dictget(fit, 'data', 'means', 'covs', 'props', 'resps', 'clust_types')  #loglik_hist = fit

    n_clust, n_var = np.array(means).shape

    #cmap = np.array([cols[ct] for ct in clust_types])
    cmap = np.array(ch.weighted_RGB(resps, ch.hex_to_RGB(ch.cluster_cols)))    
   # markers = {0: 'o', 1: 's', 2: '*'}
    #mmap = np.array(['o','o','o'])
    

    assign = np.argmax(resps.T, axis = 1)
    plt.scatter(data[:,0], data[:,1], c = cmap, zorder = 2, alpha = .05)#, marker = mmap[assign])#, marker = markers[assign])
    
    #for d, r in zip(data, resps.T):        
    #    assign = np.argmax(r)
    #    plt.plot(d[0], d[1], color = cols[assign], marker = markers[assign], alpha = .2, zorder = 2)
    
    print("")
    plot_ellipses(means, covs, clust_types)

    #abline(1, 0)
    plt.xlabel("cumulative time along midline")
    plt.ylabel("time headway of gaze")
    plt.xlim(min(data[:,0])-1, max(data[:,0])+1)
    plt.ylim(min(data[:,1])-1, max(data[:,1])+1)
    #plt.axis('equal')
    plt.title(title)
    #plt.show()
    #plt.plot(range(len(lik_hist)), lik_hist)
    #plt.title(title)
    ##plt.show()

def plot_ambiguity(fit, metric = 'sil'):

    data, means, covs, vals, clust_types = ch.dictget(fit, 'data','means','covs',metric, 'clust_types')
    plt.scatter(data[:,0], data[:,1], c = vals, cmap = 'gray_r', zorder = 2, alpha = .2)

    plot_ellipses(means[:-1], covs[:-1], clust_types[:-1])
    plt.xlim(min(data[:,0])-1, max(data[:,0])+1)
    plt.ylim(min(data[:,1])-1, max(data[:,1])+1)
    ax = plt.gca()
    ax.set_facecolor((.6, .9, .6))
    plt.title(np.nanmean(vals))


def plot_density(fit, margin = 1, title = None):
    
    data, means, covs, props, clust_types, *_ = ch.dictget(fit, 'data','means','covs','props','clust_types')
    #data = fit[0]
    th_actual = data[:,margin]
    density = gaussian_kde(th_actual)
    #print(density.covariance_factor())
    #density.covariance_factor = lambda : 
    #density._compute_covariance()
    x = np.linspace(0, max(th_actual), 500)
    ax = plt.gcf().subplots(1,2)
     

    ax[0].plot(x,density(x), 'xkcd:teal') #row=0, col=0
    ax[1].plot(x,density(x), 'xkcd:teal') #row=0, col=0
    labels = {1: 'marginal for gaze time headway', 0: 'marginal for vehicle to point'}
    plt.xlabel(labels[margin])

    #to get marginal can just drop the parameters
    totaldens = 0
    for i, (m, c, p, ct) in enumerate(zip(means, covs, props, clust_types)):
        dens = norm.pdf(x, m[margin], np.sqrt(np.diag(c))[margin])
        dens *= p
        ax[1].plot(x, dens, color = cols[ct])
        totaldens += dens
    ax[0].plot(x, totaldens, 'xkcd:magenta')
    """
    predictions, clusters = predict_gmm(fit)
    th_actual = predictions[:,1]
    density = gaussian_kde(th_actual)
    ax[0].plot(x,density(x), 'm')
    
    n_clust, _ = np.array(means).shape   
    cols = {0: 'b', 1: 'r', 2: 'g'}  
    for c in range(n_clust):
        predictions = clusters[c][:,1]                        
        if len(predictions) == 0: continue
        density = gaussian_kde(predictions)
        ax[1].plot(x,density(x)*props[c], color = cols[c])
    """

    plt.title(title)
    #plt.show()

def plot_weights_over_time(fit):

    resps = fit[5].T
   # print(resps.shape)
    obs, clusts = resps.shape
    
    
    for i in range(clusts):
        plt.plot(range(obs), resps[:,i], '-', c = cols[i], alpha = .5)

    ##plt.show()

def plot_results(fit, g):

    figs = []
    f1 = plt.figure("clusters", figsize=(6,4))
    ch.move_figure(f1, .11, .99)
    plot_clusters(fit, g)
    plt.savefig('sample_coloured_clusters.png', format='png', dpi=1200, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    figs.append(f1)
    
    f2 = plt.figure("gaze marginal", figsize = (6,4))
    ch.move_figure(f2, .11, .5)
    plot_density(fit, margin = 1, title = g)
    plt.savefig('gazemarginal_talk.png', format='png', dpi=1200, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    figs.append(f2)
    
    f3 = plt.figure("veh marginal", figsize = (6,4))
    ch.move_figure(f3, .51, .99)
    plot_density(fit, margin = 0, title = g)
    figs.append(f3)            
    
    #f4 = plt.figure("weights", figsize = (6,4))
    #move_figure(f4, .51, .5)
    #plot_weights_over_time(fit3)
    
    f5 = plt.figure("entropy", figsize = (6,4))
    ch.move_figure(f5, .51, .5)
    plot_ambiguity(fit, metric='ent')
    plt.savefig('entropy_example.png', format='png', dpi=1200, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    figs.append(f5)

    
    plt.draw() #draws all figs
    plt.pause(.01)
    input('enter to close')
    plt.close('all')

def silhouette_scores(fit, drop_noise = False):

    """The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.     The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    
    Here we ignore the noise cluster when computing the nearest cluster
    """
    
    data, means, resps = ch.dictget(fit, 'data','means','resps')
    assign = np.argmax(resps.T, axis = 1)

    if drop_noise:
        data = data[assign < 2]

    #intra_dist = np.linalg.norm(data - means[assign])
    #inter_cluster = np.empty(len(data))
    sample_scores = np.empty(len(data))
    
    for i, (a, d) in enumerate(zip(assign, data)):
        
        a_dist = np.linalg.norm(d - means[a])
        
        if a == 0: b = 1
        if a == 1: b = 0

        if (a == 2) and (not drop_noise):    
            b_dist = min(np.linalg.norm(d - means[0]), np.linalg.norm(d - means[1]))
        else:
            b_dist = np.linalg.norm(d - means[b])
                
        #a_dist = intra_dist[i]
        sample_scores[i] =  (b_dist - a_dist) / max(a_dist, b_dist)

    return(sample_scores)

def entropy_scores(fit):

    """shannons entropy"""
    resps = fit['resps']
    entropy = np.array([-np.sum(r * np.log(r))
    for r in resps.T])
   # print(entropy.shape)
    return(entropy)
    


def predict_gmm(fit):
     
    data, means, covs, props = ch.dictget(fit, 'data','means','covs','props')
    n_obs, n_var = data.shape
    n_clust, _ = np.array(means).shape     

    X = []
    for m, c, p in zip(means, covs, props):
        X.append(multivariate_normal.rvs(mean=m, cov=c, size=int(p*n_obs)))

    X_predict = np.concatenate(X)

    return(X_predict, X)

def initial_values(gm_cumdist, cd_info, clusters = ['gf','entry','exit']):

    mn, var = cd_info
    n_clust = len(clusters)
    props = np.full(n_clust, 1/n_clust) #this has an array of size n_clust summing to one, equal proportions.

    means = {'gf': [mn, 2.0], 'entry':[gm_cumdist[0], 4.0], 'exit':[gm_cumdist[1], 4.0]}
    covs = {'gf': [[var, 0],
            [0, 2.0]], 'entry':[[4.0, -3],
            [-3, 4.0]], 'exit':[[4.0, -3],
            [-3, 4.0]]}

    true_means = []
    true_covs = []
    for c in clusters:
        true_means.append(means[c])
        true_covs.append(covs[c])
    
    return(true_means, true_covs, props, n_clust)

def run_simulated_gmm():

    """old - will not work"""
    
    data = simulate_data()

    cols = {0: 'b', 1: 'r', 2: 'g'}  
    clusters = {0: 'gf', 1: 'laf', 2:'noise'}

    means, covs, props, n_clust = initial_values()
    plot_ellipses(means, covs)
    abline(1,0)
    #plt.show()
    
    fit2 = gmm_em(data, means, covs, props, n_clust, noise = False)
    plot_clusters(fit2)
    plot_density(fit2)
    #predict_gmm(fit2)

    fit_noise = gmm_em(data, means, covs, props, n_clust, noise = True)
    plot_clusters(fit_noise)
    plot_density(fit_noise)

def get_gazemodes_cumdist(gm):

    ml = ch.get_midline()
    tree = ch.get_tree(ml)
    cumdist = ch.get_cumdist(ml)    
    _, closest = tree.query(gm)
    gm_cumdist = cumdist[closest]

    return(gm_cumdist)


def run_real_gmm(plot = True):

    steergaze_df = pd.read_feather('../Data/trout_2.feather')
    steergaze_df = steergaze_df.query('roadsection < 2').copy() #not slalom.

    #find cumdist of gaze mode points
    gazemodes = np.array([[-22.74,70.4],[25,50.39]])
    gm_cumdist = get_gazemodes_cumdist(gazemodes)
    gm_cumdist /= 8.0 #convert to time
    gm_cumdist -= 4.0 #minus some average laf time headway
    print(gm_cumdist)    

    #steergaze_df = pd.read_csv('../Data/segmentation_scaled_101019.csv')
    columns = ['clust_n','clust_type','clust_veh_mean','clust_th_mean','clust_veh_var','clust_th_var','clust_cov','clust_prop','entropy','drivingmode','ID']
    output = []

    cd_info = [np.mean(steergaze_df.midline_cumdist.values / 8.0), np.var(steergaze_df.midline_cumdist.values / 8.0)]
    
    drivingmodes = [0,1,2]
    for drive in drivingmodes:
                
        #query_string = "drivingmode < 2 & roadsection == 1 & sample_class != 2 & confidence > .8"     
        query_string = "drivingmode == {} & confidence > .8".format(drive)     
        print(query_string)
        data = steergaze_df.query(query_string).copy()
        #data.sort_values('currtime', inplace=True)

        """
        def arc_length(th_array, r = 25, sp = 8):

            r *= 2
            th_array *= sp #convert to distances    
            arc_array = r * np.arcsin(th_array/r)
            arc_array /= 8
            #print(arc_array)
            
            return(arc_array)
        """
        #correct for typo
        data.loc[data.ID == 203, 'ID'] = 503        

        for g, d in data.groupby(['ID']): 
            
            cumdist = d.midline_cumdist.values / 8.0   
            th_actual = d.th_along_midline.values

            """
            density = gaussian_kde(th_actual)
            #print(density.covariance_factor())
            #density.covariance_factor = lambda : 
            #density._compute_covariance()
            x = np.linspace(0, max(th_actual), 500)
            #]ax = plt.gcf().subplots(1,2)
            
            ax = plt.subplot(111)
            ax.plot(x,density(x), ch.raw_col, linewidth = 1.5) #row=0, col=0
            ax.set_ylabel('Density', fontsize = 16)
            ax.set_xlabel('Gaze Time Headway (s)', fontsize = 16)
            ax.set_xlim(0, max(th_actual))

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')    

            plt.savefig('timeheadway_marginal.png', format='png', dpi=1200, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
            plt.show()

            hehe
            """
            
            
            X = np.transpose(np.array([cumdist, th_actual]))
            X = X[~np.isnan(X).any(axis=1)].reshape(-1,2)
                        
            clust_types = ['gf','entry','exit']
            means, covs, props, n_clust = initial_values(gm_cumdist, cd_info, clusters = clust_types)
            
            fit = gmm_em(X, means, covs, props, n_clust, clust_types, noise = True, plot=False)

            #fit['sil'] = silhouette_scores(fit)
            fit['ent'] = entropy_scores(fit)
            #print(sil_scores)
            print(fit['covs'])

            if plot:
                plot_results(fit, g)

            means, covs, props, entropy, clust_types = ch.dictget(fit, 'means','covs','props','ent','clust_types')
           # sil = np.nanmean(sil_score)
            ent = np.nanmean(entropy)

            for i, (m, cov, p, ct) in enumerate(zip(means, covs, props, clust_types)):
                out = [i, ct, *m, *np.diag(cov),cov[0][1], p, ent, drive, g]
                output.append(out)

            

    res = pd.DataFrame(output, columns = columns)
    res.to_csv("gmm_res.csv")                                    
    return (res)      

if __name__ == '__main__':

    #run_simulated_gmm()
    run_real_gmm(plot=True)