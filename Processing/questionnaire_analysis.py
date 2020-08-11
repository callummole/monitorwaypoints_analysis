import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Everyone completed this questionnaire after the experiment:

1) How human-like did you find the automated steering control?
'Not at all human-like 0 1 2 3 4 5 6 7 Human-like'

2) During automation, how ready did you feel to immediately take over the vehicle?'
'Not at all ready 0 1 2 3 4 5 6 7 Ready'

3) During automation, to what extent were you looking in the same place as when you were driving?
'Not at all in the same place 0 1 2 3 4 5 6 7 In the same place'

4) How similar to your own steering behaviour did you feel the automated vehicle was? 
Not at all similar 0 1 2 3 4 5 6 7 very similar
"""
qus = {0: 'How human-like did you find the automated steering control?',
 1: 'During automation, how ready did you feel to immediately take over the vehicle?',
 2: 'During automation, to what extent were you looking in the same place as when you were driving?',
 3: 'How similar to your own steering behaviour did you feel the automated vehicle was?'}

d1 = pd.read_csv("../Data/PostTest.csv")
d2 = pd.read_csv("../Data/PostTest_rerun.csv")
d1['dataset'] = 1
d2['dataset'] = 2
D = pd.concat([d1, d2])
D.reset_index(inplace = True)
D.drop([25], axis = 0, inplace = True)

labels = ['q1','q2','q3','q4']
for i,l in enumerate(labels):
    ans = D[l].values
    print("\n")
    print(qus[i])
    print("mean", np.mean(ans))
    print("sd", np.sqrt(np.var(ans)))

    print("mean", np.mean(ans) / 8 * 100)
    print("sd", np.sqrt(np.var(ans)) / 8 * 100)

    print(range(8) )
    print( np.array(range(8)) /8 * 100)
    plt.hist(ans)
    plt.title(qus[i])
    plt.show()



