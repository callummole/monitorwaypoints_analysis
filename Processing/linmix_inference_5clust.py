import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linmix as lm
from scipy.stats import norm, t, pearsonr, spearmanr
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import polyfit
import calhelper as ch



def load_data(rerun = True, plot = False):
    
    if rerun:
        res = lm.main(plot=plot)
    else:
        res = pd.read_csv("linmix_res_5clust.csv")   
        #res = res.query('dataset == 2').copy()

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

    """plots multiplot inter-individual correlations across driving mode for time headway and weights"""

    #time headway, weights, variances
    fig, ax = plt.subplots(3,3, figsize=(15,15),  num = "correlations")

    titles = ['Time Headway Mean','Time Headway Mode', 'Proportions']
    xlabs = ['active mean th (s)','active mode (s)', 'active weight (%)']
    ylabs = ['passive mean th (s)','passive mode (s)', 'passive weight (%)']
    
    datacolumns = ['mean','mode','weight']
    noise = max(res.clust_n.values)

    for ci, (clust, d) in enumerate(res.groupby(['clust_n'])):
        
        if clust > 2: continue
        
        active = d.query('drivingmode == 0').copy()
        active.sort_values('ID',inplace=True)
        passive = d.query('drivingmode == 1').copy()
        passive.sort_values('ID',inplace=True)

        for di, dc in enumerate(datacolumns):
            act = np.array(active.loc[:,dc])
            pas = np.array(passive.loc[:,dc])
            dataset = active.dataset.values

            r,p = pearsonr(act,pas)

            itc, slo = polyfit(act, pas, 1)

            axes = ax[ci, di]

            axes.plot(act, pas, 'o', color = ch.cluster_cols[clust])
            axes.text(.7, .2, 'r: ' + str(round(r,2)), transform = axes.transAxes)
            axes.text(.7, .1, 'p: ' + str(round(p,2)), transform = axes.transAxes)
            axes.plot(act, itc + slo * act, 'k-')
            abline(1, 0, axes)
            axes.set(xlabel = xlabs[di], ylabel = ylabs[di])
    
    for a, ct in zip(ax[0], titles):
        a.set_title(ct)
    
    rowtls = ['GF','Entry','Exit']
    for a, rt in zip(ax[:,0], rowtls):
        a.annotate(rt, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 5, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation = 90)
            
    #plt.savefig('corr_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()



def CIs(d, ci = .99):

    m = np.nanmean(d)
    n = len(d)
    sd = np.sqrt (np.sum( (d - m)**2 ) / (n-1))
    se = sd / np.sqrt(n)
    cis = se * t.ppf( (1+ci) / 2, n-1)  
      
    return(m, cis)

def plot_within(res):
    """plots within participant differences"""

    #time headway, weights, variances
    fig, ax = plt.subplots(1,3, figsize=(10,6),  num = "within participant differences")

    titles = ['Time Headway Mean','Time Headway Mode', 'Proportions']
    
    ylabs = ['Passive - Active mean th (s)','Passive - Active mode (s)', 'Passive - Active weight (%)']
    
    datacolumns = ['mean','mode','weight']
    noise = max(res.clust_n.values)

    for i, (clust, d) in enumerate(res.groupby(['clust_n'])):
        
        if clust > 2: continue
        
        active = d.query('drivingmode == 0').copy()
        active.sort_values('ID',inplace=True)
        passive = d.query('drivingmode == 1').copy()
        passive.sort_values('ID',inplace=True)


        for di, dc in enumerate(datacolumns):
            act = np.array(active.loc[:,dc])
            pas = np.array(passive.loc[:,dc])
            dataset = active.dataset.values

            diff = pas-act
            m, ci= CIs(diff, ci = .95)

            axes = ax[di]
                        
            axes.errorbar((clust*.3) + .5, m, yerr = ci, c = ch.cluster_cols[clust])
            axes.plot((clust*.3) + .5, m, c = ch.cluster_cols[clust], marker = 'o')
            axes.set_title(titles[di])
            #ax.flat[i].set(xlabel = 'Road Section', ylabel = ylabs[i])
            axes.set_xticks([]) 
            axes.set_xticklabels([''])  
            axes.axhline(y=0.0, color=(.4,.4,.4), linestyle='--')
    
    legend_elements = [Line2D([0], [0], color = 'xkcd:blue', lw=4, label='GF'),
                        Line2D([0], [0], color = 'xkcd:red', lw=4, label='Entry'),
                        Line2D([0], [0], color = 'xkcd:orange', lw=4, label='Exit')                       
                   ]
    ax.flat[2].legend(handles=legend_elements, loc = [.4,.1])
    #plt.savefig('corr_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()

def plot_pointranges(res):
    """plots a multiplot cluster size, mean_th etc"""
    
    markers = {0: 'o', 1: 's', 2:'*'}
    al = {0: 1, 1: .5, 2:.5}
    
    fig, ax = plt.subplots(1,3, figsize=(10,6),  num = "point estimates")

    titles = ['Mean Time Headway','Mode Time Headway', 'Cluster Proportions']
    ylabs = ['mean th (s)', 'mode th (s)', 'mean weight (%)']
    
    noise = max(res.clust_n.values)
    for g, d in res.groupby(['drivingmode','clust_n']):        
        drive, clust_n = g[0], g[1]
        if clust_n > 2:continue #noise
        datlist = [d['mode'].values,
        d['mean'].values,
        d['weight'].values]

        clust_n = d.clust_n.values[0]         
        
        for i, arr in enumerate(datlist):

            #print(titles[i])
            m, ci= CIs(arr, ci = .95)
            #print('mean', m)
            #print('CI', ci)
            ax.flat[i].errorbar((clust_n*.3)+(drive*.1), m, yerr = ci, c = ch.cluster_cols[clust_n], alpha = al[drive])
            ax.flat[i].plot((clust_n*.3)+(drive*.1), m, c = ch.cluster_cols[clust_n], marker = markers[drive], alpha = al[drive])
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
    #plt.savefig('cluster_pointranges_linmix.png', format='png', dpi=600, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
    plt.show()

res = load_data(rerun = True, plot = False)
#plot analytic marginal cluster distributions in four subplots

plot_pointranges(res)

plot_corr(res)

plot_within(res)