import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 
import numpy as np
from scipy.stats import multivariate_normal
import feather
import pandas as pd
from pprint import pprint
from GMM_class import GMM


from matplotlib.patches import Ellipse
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2, alpha = .05)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2, alpha = .05)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


steergaze_df = pd.read_feather('../Data/trout_rerun.feather')

#only for manual bends to begin with.
query_string = "drivingmode < 2 & roadsection == 1 & confidence > .6"     
data = steergaze_df.query(query_string).copy()
data.sort_values('currtime', inplace=True)

def arc_length(th_array, r = 25, sp = 8):

    r *= 2
    th_array *= sp #convert to distances    
    arc_array = r * np.arcsin(th_array/r)
    arc_array /= 8
    #print(arc_array)
    return(arc_array)

for g, d in data.groupby(['ID','block']):

    th_to_target = arc_length(d.th_to_target.values)
    th_actual = arc_length(d.timeheadway_angularref.values)
    
    X = np.transpose(np.array([th_to_target, th_actual]))
    X = X[~np.isnan(X).any(axis=1)].reshape(-1,2)
    
    #get rough values for variance and covariance for the two clusters.
    gf_lb = 3.5
    gf_data = X[X[:,0]>gf_lb,:]
    cov_gf = np.cov(gf_data.T)

    gf_len = len(gf_data)

    print("\n")
    print("GF_cov", cov_gf)
    print("GF_means", np.mean(gf_data, axis = 0))
    
    laf_data = X[X[:,0]<gf_lb,:]
    cov_laf = np.cov(laf_data.T)

    laf_len = len(laf_data)
    proportions = np.array([gf_len, laf_len])
    print("proportions", proportions)

    mysum = np.sum(proportions, axis = 0)
    proportions = proportions / mysum
    print("proportions", proportions)
    
    print("laf_cov", cov_laf)
    print("laf_means", np.mean(laf_data, axis = 0))

    
    #print("GF mmeans:", np.mean(gf_targets), np.mean(gf_th))
    
    
    
    #plt.plot(X[:,1], X[:,0], 'o', alpha = .05)
    

   # gmm = GaussianMixture(n_components=2, covariance_type='full')
   # gmm_fit = gmm.fit(X)
    #plot_gmm(gmm, X)
    #use class developed by https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f

    #model = GMM(2, n_runs = 5)
    #fitted_values = model.fit(X)
    #predicted_values = model.predict(X)

    
    #plt.scatter(X[:, 0], X[:, 1],c=predicted_values ,s=20, cmap='viridis', zorder=2, alpha = .1)
    plt.scatter(X[:, 0], X[:, 1],s=20, cmap='viridis', zorder=2, alpha = .1)
    plt.axvline(gf_lb)
    #w_factor = 0.2 / model.pi.max()
    #for pos, covar, w in zip(model.mu, model.sigma, model.pi):
    #    draw_ellipse(pos, covar, alpha = w)
    plt.xlabel("th_to_object")
    plt.ylabel("th_actual")
    plt.show()


