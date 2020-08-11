import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#rootdir = "E:/Trout_rerun_gazevideos" 
rootdir = "C:/git_repos/sample_gaze_Trout"  

gazedata_filename = 'gaze_on_surface.csv'
    #print(pfolder)
#print ("here")
for dirs in os.walk(rootdir):
    path = str(dirs[0]) 
    print (path)
    
    gaze_path = os.path.join(path,gazedata_filename)
    if os.path.exists(gaze_path):   
        


        timestamps = np.load(os.path.join(path, "world_timestamps.npy"))        

        gazedata = pd.read_csv(gaze_path)

        world_ts = gazedata['world_timestamp'].values #video
        gaze_ts = gazedata['gaze_timestamp'].values #pupil
        recv_ts = gazedata['recv_timestamp'].values #vizard 
        #plt.plot(timestamps)

        #plt.plot(np.diff(gaze_ts))
        plt.plot(recv_ts, gaze_ts - recv_ts)

        plt.show()
        
