import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmm_2d_cumdist as gmm_2d
from scipy.stats import norm, t, pearsonr, spearmanr
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import polyfit

cluster_col = {'gf': 'xkcd:blue', 'entry': 'xkcd:red', 'exit':'xkcd:orange', 'noise': 'xkcd:green'}  

def load_data(rerun = True, plot = False):
    
    if rerun:
        res = gmm_2d.run_real_gmm(plot=plot)
    else:
        res = pd.read_csv("gmm_res.csv")   

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

def plot_corr(res):

    """plots multiplot inter-individual correlations across driving mode for time headway and weights and variances"""

    #time headway, weights, variances
    fig, ax = plt.subplots(3,3, figsize=(15,15))

    titles = ['Time Headway','Variances', 'Proportions']
    xlabs = ['active mean th (s)','active variance', 'active weight (%)']
    ylabs = ['passive mean th (s)','passive variance', 'passive weight (%)']
    
    datacolumns = ['clust_th_mean','clust_th_var','clust_prop']

    for ci, (clust, d) in enumerate(res.groupby(['clust_type'])):
        
        if clust == 'noise': continue
        

        active = d.query('drivingmode == 0').copy()
        active.sort_values('ID',inplace=True)
        passive = d.query('drivingmode == 1').copy()
        passive.sort_values('ID',inplace=True)

        for di, dc in enumerate(datacolumns):
            act = np.array(active.loc[:,dc])
            pas = np.array(passive.loc[:,dc])

            r,p = spearmanr(act,pas)

            itc, slo = polyfit(act, pas, 1)

            axes = ax[ci, di]

            axes.plot(act, pas, 'o', color = cluster_col[clust])
            axes.text(.7, .2, 'r: ' + str(round(r,2)), transform = axes.transAxes)
            axes.text(.7, .1, 'p: ' + str(round(p,2)), transform = axes.transAxes)
            axes.plot(act, itc + slo * act, 'k-')
            abline(1, 0, axes)
            axes.set(xlabel = xlabs[di], ylabel = ylabs[di])
    
    for a, ct in zip(ax[0], titles):
        a.set_title(ct)
    
    rowtls = ['Entry','Exit','GF']
    for a, rt in zip(ax[:,0], rowtls):
        a.annotate(rt, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation = 90)
            
    plt.show()


def CIs(d, ci = .99):

    m = np.nanmean(d)
    n = len(d)
    sd = np.sqrt (np.sum( (d - m)**2 ) / (n-1))
    se = sd / np.sqrt(n)
    cis = se * t.ppf( (1+ci) / 2, n-1)  
      
    return(m, cis)


def plot_pointranges(res):
    """plots a multiplot cluster size, mean_th etc"""
    
    markers = {0: 'o', 1: 's', 2:'*'}
    al = {0: 1, 1: .5, 2:.5}
    #subplots of th_mean, th_var, clust_prop, entropy
    fig, ax = plt.subplots(2,2, figsize=(12,10))

    titles = ['Mean Time Headway','Time Headway Variances', 'Cluster Proportions', 'Entropy']
    ylabs = ['mean th (s)','mean variance', 'mean weight (%)', 'mean entropy (nat)']
    
    for g, d in res.groupby(['drivingmode','clust_type']):

        drive, clust_type = g[0], g[1]
        if clust_type == "noise":continue
        datlist = [d.clust_th_mean.values,
        d.clust_th_var.values,
        d.clust_prop.values,
        d.entropy.values]

        clust_n = d.clust_n.values[0]         
        
        for i, arr in enumerate(datlist):
            m, ci= CIs(arr, ci = .95)
            ax.flat[i].errorbar((clust_n*.3)+(drive*.1), m, yerr = ci, c = cluster_col[clust_type], alpha = al[drive])
            ax.flat[i].plot((clust_n*.3)+(drive*.1), m, c = cluster_col[clust_type], marker = markers[drive], alpha = al[drive])
            ax.flat[i].set_title(titles[i])
            #ax.flat[i].set(xlabel = 'Road Section', ylabel = ylabs[i])
            ax.flat[i].set_xticks([]) 
            ax.flat[i].set_xticklabels([''])        
    
    legend_elements = [Line2D([0], [0], color = 'xkcd:blue', lw=4, label='GF'),
                        Line2D([0], [0], color = 'xkcd:red', lw=4, label='Entry'),
                        Line2D([0], [0], color = 'xkcd:orange', lw=4, label='Exit'),
                        Line2D([0], [0], marker='o', color='w', label='Active',
                          alpha =1, markerfacecolor='k'),
                        Line2D([0], [0], marker='s', color='w', label='Passive',
                          alpha =.5, markerfacecolor='k'),
                          Line2D([0], [0], marker='*', color='w', label='Stock',
                          alpha =.5, markerfacecolor='k')
                   ]
    ax.flat[2].legend(handles=legend_elements, loc = [.4,.6])
    plt.savefig('cluster_pointranges.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()
    

def plot_marginals(res, plottype = 'id'):

    """plottype:
    id = id marginals
    id_e = id marginals with entropy alpha
    group = condition marginals
    """
    filenames = {'id': 'participant_marginals.png',
    'id_e':'participant_marginals_entropy.png',
    'group':'condition_marginals.png'
    }

    if plottype not in filenames.keys():
        raise Exception('unrecognised plottype')

    groupby = []
    if 'id' in plottype: groupby = ['ID']


    x = np.linspace(0, 7, 200)
    ax = plt.gcf().subplots(3,1, sharex=True, sharey=True)
    labels = {1: 'Marginal for Gaze Land Point Time headway', 0: 'marginal for vehicle to point'}        
    #columns = ['clust_n','clust_type','clust_veh_mean','clust_th_mean','clust_veh_var','clust_th_var','clust_prop','sil_score','entropy','roadsection','drivingmode','ID']
    
    #groupby.append('lafsection')
    groupby.append('drivingmode')
    groupby.append('clust_type')
    
    for g, d in res.groupby(groupby):
        
        totaldens = 0
        totalmean = 0
        totalvar = 0
        for i, (_, row) in enumerate(d.iterrows()):
            m, var, p, e = row.clust_th_mean, row.clust_th_var, row.clust_prop, row.entropy
            clust_type, drive = row.clust_type, row.drivingmode
            
            dens = norm.pdf(x, m, np.sqrt(var))
            dens *= p
            totaldens += dens            

        totaldens /= i+1 #average

        al = {'id': .3, 'id_e': e*3, 'group':1}
        ax[drive].plot(x, totaldens, color = cluster_col[clust_type], alpha = al[plottype])

        
    plt.suptitle(labels[1])
    #plt.ylim(0,2)
    #plt.xlim(0,7)
    
    for a in ax.flat:
        a.set(xlabel='Gaze Time Headway', ylabel='')    

    #coltls = ['Approach','Bends','Exit']
    #for a, ct in zip(ax[0], coltls):
     #   a.set_title(ct)
    
    rowtls = ['Active','Passive','Stock']
    for a, rt in zip(ax, rowtls):
        a.annotate(rt, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation = 90)

    print(filenames[plottype]) 
    plt.savefig(filenames[plottype], format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')


res = load_data(rerun = True, plot = False)
#plot analytic marginal cluster distributions in four subplots
"""
fig = plt.figure("Marginals")
plot_marginals(res, 'group')
plt.show()

fig = plt.figure("Marginals")
plot_marginals(res, 'id')
plt.show()

fig = plt.figure("Marginals")
plot_marginals(res, 'id_e')
plt.show()
"""

#plot_pointranges(res)


plot_corr(res)