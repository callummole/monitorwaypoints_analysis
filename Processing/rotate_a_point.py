import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



if __name__ == '__main__':
	#rootdir = sys.argv[1] 
    
    #5.907457, 2.47021
    #point = 4, 5
    point = 5.907457, 2.47021
    angle = .5 #rads. 45 degrees            
                                                                   
    #rotate point of gaze using heading angle.
    xrotated = (point[0] * np.cos(angle)) - (point[1] * np.sin(angle))
    zrotated = (point[0] * np.sin(angle)) + (point[1] * np.cos(angle))

    print(xrotated, zrotated)

    plt.plot(point[0],point[1], "bo")
    plt.plot(xrotated,zrotated, "ro")
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()




