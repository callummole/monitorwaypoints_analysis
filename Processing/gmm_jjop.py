import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import softmax, logsumexp

def gmm(data, means, covs, props, niter=1000, tol=1e-6):
    C = len(means)
    N = len(data)
    means = np.copy(means)
    covs = np.copy(covs)
    props = np.copy(props)
    prevlik = None
    lik_history = []
    liks = np.zeros((len(means), len(data)))
    for i in range(niter):
        for c in range(C):
            lik = multivariate_normal.logpdf(data, means[c], covs[c]) + np.log(props[c])
            liks[c] = lik
            
        total = np.sum(logsumexp(liks, axis=0))
        lik_history.append(total)
        
        if i > 0:
            change = total - lik_history[-2]
            assert(change > 0)
            if change < tol:
                break
        
        resps = softmax(liks, axis=0)
        totals = np.sum(resps, axis=1)
        for c in range(C):
            m = np.dot(resps[c], data)/totals[c]
            means[c] = m
            diff = data - m
            covs[c] = np.dot(resps[c]*diff.T, diff)/totals[c]
            props[c] = totals[c]/N
        

        
    return means, covs, props, lik_history

def test():
    import matplotlib.pyplot as plt
    N = 1000
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    covs = np.array([[[1.0, 0.1], [0.1, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])
    props = np.array([1.0, 2.0])
    props /= np.sum(props)

    data = []
    for m, c, p in zip(means, covs, props):
        data.append(multivariate_normal.rvs(mean=m, cov=c, size=int(p*N)))
    data = np.concatenate(data)
    
    print(means)
    means += np.random.randn(*means.shape)*2
    means, covs, props, steps = gmm(data, means, covs, props)
    print(means)
    #for m, c, p in zip(means, covs, props):
    #    plt.plot(rng, multivariate_normal.pdf(rng, m, c)*p)
    plt.figure("steps")
    plt.plot(steps)
    plt.show()

if __name__ == '__main__':
    test()
            

