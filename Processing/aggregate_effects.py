import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import t, stats
from matplotlib.lines import Line2D

df = pd.read_feather('../Data/trout_subset_2.feather')

query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
df = df.query(query_string).copy()

IDs = list(set(df.ID.values))
means = np.empty([len(IDs), 3])
meds = np.empty([len(IDs), 3])
for g, d in df.groupby(['ID','drivingmode']):

    th_actual = d.th_along_midline.values
    #exclude datapoints that are very far away from the midline.
    exclude_crit = 20 #20 degrees
    dists = np.sqrt(d.midline_vangle_dist.values ** 2 + d.midline_hangle_dist.values ** 2)
    include = dists<exclude_crit

    th_actual = th_actual[include]

    
    ppid, dm = int(g[0]), int(g[1])
    ppidx = IDs.index(ppid)	

    mn = np.mean(th_actual)
    means[ppidx, dm] = mn
    med = np.median(th_actual)
    meds[ppidx, dm] = med
    
def CIs(d, ci = .99):

    m = np.nanmean(d)
    n = len(d)
    sd = np.sqrt (np.sum( (d - m)**2 ) / (n-1))
    se = sd / np.sqrt(n)
    cis = se * t.ppf( (1+ci) / 2, n-1)  
      
    return(m, cis)

def plot_pointranges(means):
    """plots a multiplot cluster size, mean_th etc"""
    
    markers = {0: 'o', 1: 's', 2:'*'}
    al = {0: 1, 1: .5, 2:.5}
    
    titles = ['Time Headway']

    fig, ax = plt.subplots(1,len(titles), figsize=(6,6),  num = "point estimates")
    ylabs = ['mean th (s)']
    
    ID, dms = means.shape
    for drive in range(dms):        
        
        arr = means[:,drive]
        #print(titles[i])
        m, ci= CIs(arr, ci = .95)

        print(g)        
        print('mean', m)
        print('CI', m - ci, m + ci)
        print("\n")

        ax.errorbar((drive*.1), m, yerr = ci, alpha = al[drive])
        ax.plot((drive*.1), m, marker = markers[drive], alpha = al[drive])
        ax.set_title(titles[0])        
        ax.set_xticks([]) 
        ax.set_xticklabels([''])        
        ax.set_ylim(ymin = 1.5, ymax = 3.5)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Active',
                          alpha =1, markerfacecolor='k'),
                        Line2D([0], [0], marker='s', color='w', label='Passive',
                          alpha =.5, markerfacecolor='k'),
                          Line2D([0], [0], marker='*', color='w', label='Stock',
                          alpha =.5, markerfacecolor='k')
                   ]
    ax.legend(handles=legend_elements, loc = [.6,.6])
    #ax.flat[0].set_ylim(ymin=1.5, ymax = 4)
    #ax.flat[-1].set_ylim(ymin = 0, ymax = 1)
    #plt.savefig('cluster_pointranges_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()

def plot_hists(res, pairs = [[0,1],[0,2],[1,2]]):
    """plots within participant differences"""

    #time headway, weights, variances
    fig, ax = plt.subplots(1,len(pairs), figsize=(6,6), num = "within participant differences")


    for i, drives in enumerate(pairs):
        
        arr1 = res[:, drives[0]]
        arr2 = res[:, drives[1]]
        
        diff = arr2-arr1
        ax[i].hist(diff)                
        
        
    #plt.savefig('withindiffs_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()

def plot_within(res, pairs = [[0,1],[0,2],[1,2]]):
    """plots within participant differences"""

    #time headway, weights, variances
    fig, ax = plt.subplots(1,1, figsize=(6,6), num = "within participant differences")

    titles = ['Time Headway Mean','Time Headway Mode', 'Proportions']
    
    ylabs = ['Passive - Active (s)','Stock - Active (s)', 'Stock - Passive (s)']
    

    for i, drives in enumerate(pairs):
        
        arr1 = res[:, drives[0]]
        arr2 = res[:, drives[1]]
        
        diff = arr2-arr1

        m, ci= CIs(diff, ci = .95)

        

        t, p = stats.ttest_1samp(diff,0)
                        
        ax.errorbar((i*.3) + .5, m, yerr = ci)
        ax.plot((i*.3) + .5, m, marker = 'o')
        #ax.set_title(titles[di])
        #ax.flat[i].set(xlabel = 'Road Section', ylabel = ylabs[i])
        ax.text((i*.35 + .1), .2, 't: ' + '% .2g'%t, transform = ax.transAxes)
        ax.text((i*.35 + .1), .15, 'p: ' + '% .2g'%p, transform = ax.transAxes)
        ax.set_xticks([.5, (1*.3) + .5, (2*.3) + .5]) 
        ax.set_xticklabels(ylabs)  
        ax.axhline(y=0.0, color=(.4,.4,.4), linestyle='--')
        
    #plt.savefig('withindiffs_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()
#
#plot_hists(means)
#plot_pointranges(means)
plot_within(means)

plot_pointranges(meds)
plot_within(meds)