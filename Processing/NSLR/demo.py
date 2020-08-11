import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from nslr_hmm import *

t = np.arange(0, 5, 0.01)
saw = ((t*10)%10)/10.0*10.0 # 10 deg/second sawtooth
eye = np.vstack(( saw, -saw )).T #why are there two columns? 

sample_class, segmentation, seg_class = classify_gaze(t, eye)

COLORS = {
        FIXATION: 'blue',
        SACCADE: 'black',
        SMOOTH_PURSUIT: 'green',
        PSO: 'yellow',
}

eye += np.random.randn(*eye.shape)*0.1
plt.plot(t, eye[:,0], '.')

for i, seg in enumerate(segmentation.segments):
    cls = seg_class[i]
    plt.plot(seg.t, np.array(seg.x)[:,0], color=COLORS[cls])
    
plt.show()
