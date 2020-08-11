import matplotlib.pyplot as plt
import numpy as np

def entropy(weights):

    """shannons entropy"""

    #base change rule https://stackoverflow.com/questions/25169297/numpy-logarithm-with-base-n
        
    entropy = np.array([-np.sum(w * (np.log(w) / np.log(len(w)))) for w in weights])
    return(entropy)
    
def make_scores(diffs):

    total = 1.0        
    w1 = (total-diffs) / 2.0
    w2 = total-w1

    return(np.array([w1,w2]).T)

diffs = np.linspace(.01,.99,20)
#plt.plot(diffs, - (np.log(diffs) * diffs))
#plt.show()

scores = make_scores(diffs)
entropy = entropy(scores)


plt.plot(diffs,entropy)
plt.xlabel("difference between weights")
plt.ylabel("entropy")
plt.show()