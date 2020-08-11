import pandas as pd 
import matplotlib.pyplot as plt 


res = pd.read_csv("linmix_by_block_2.csv")   

res = res.query('clust_n == 0').copy()
mn = res['mean'].values

plt.hist(mn, bins = 15)

plt.show()


"""
Cluster inference:

We have distribution means for each driving mode, pp, and block.

To estimate the population means we don't particularly need the spread (since this is the spread of the response measurement). We can just have the means. 
I'm more interested at this stage in the distribution of the means. This could be a multi-level model with a separate deflection due to driving mode and block.

This is the same as the subtraction.


TODO: 

bayesian estimation of means, to enable subtractions.

bayesian correlations.
http://www.sumsar.net/blog/2013/08/bayesian-estimation-of-correlation/
https://medium.com/@ph_singer/bayesian-correlation-with-pymc-5dc6403e0599


"""