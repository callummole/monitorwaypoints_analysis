import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


datafolder = '../Data/CalibrationData'

accuracy = np.empty([25, 2]) #holds pps accuracy tests
calibration = np.empty([25, 2]) #holds pps calibration tests
i = 0
for dirs in os.walk(datafolder):        
        
    path = str(dirs[0])
    #print(path)
    

    if 'P' in path:
        ppid = path[path.find('P')+1:]
        #print(ppid)

        
        #accuracy test
        for block in [1,2]:

           # print(block)
            fname = f'Trout_{ppid}_{block}_accuracy_test.csv'
            data = pd.read_csv(path + '/' + fname, header=None)            
            acc = data[0][0]
            l = len('calibration.')
            acc = acc[l:]
            accuracy[i,block-1] = acc
            #print(acc)            

            fname = f'Trout_{ppid}_{block}_calibration_accuracy.csv'
            data = pd.read_csv(path + '/' + fname, header=None)
            
            cal = data.iloc[-1,0] #there may be multiple
            calibration[i,block-1] = cal
            #print(cal)
            
        i +=1

#plt.hist(calibration[:], alpha = .5)
#plt.hist(accuracy[:], alpha = .5)
#plt.show()

#plot alongside time headway mean
res = pd.read_csv("linmix_res_5.csv")
#res = res.query('ID != 4').copy()
res = res.query('dataset == 1 & drivingmode == 0 & clust_n == 0').copy()

plt.figure()
mnclbs = np.empty(25)
gfs = np.empty(25)
for pp, d in res.groupby(['ID']):
    
    #mnclb = np.mean(calibration[int(pp)-1, :])
    mnclb = np.mean(accuracy[int(pp)-1, :])
    gf = d['mean'].values
    
    mnclbs[int(pp)-1] = mnclb
    gfs[int(pp)-1] = gf


r,p = pearsonr(gfs,mnclbs)
plt.scatter(gfs, mnclbs)
axes = plt.gca()
axes.text(.7, .2, 'r: ' + str(round(r,2)), transform = axes.transAxes)
axes.text(.7, .1, 'p: ' + str(round(p,2)), transform = axes.transAxes)
#axes.set(xlabel = 'Time Headway Mean (s)', ylabel = 'Calibration accuracy (degs)')
axes.set(xlabel = 'Time Headway Mean (s)', ylabel = 'Accuracy Test accuracy (degs)')
plt.savefig('old_dataset_accuracytest.png', format='png', dpi=300, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
plt.show()



"""
plt.plot(calibration[:,0], calibration[:,1], 'bo', alpha = .8)
plt.plot(accuracy[:,0], accuracy[:,1], 'ro', alpha = .8)
plt.show()
"""

"""
mylist = [calibration, accuracy]
myvals = ['calibration', 'accuracy']
for val, name in zip(mylist, myvals):
    print("\n", name)
    pp_calib = np.mean(val, axis=1)
    print("mean", np.mean(pp_calib))
    print("median", np.median(pp_calib))
    print("sd", np.sqrt(np.var(pp_calib)))
    print("range", max(pp_calib)-min(pp_calib))
    #plt.hist(pp_calib, bins = 10)
    #plt.show()

out = pd.DataFrame(calibration)
out.to_csv("out.csv")
"""

#pp_calib = np.mean(calibration, axis=1)


    #if folder_skip in path: continue
    #if os.path.exists(path + gazefile):
        
        #if os.path.exists(path+ "gaze_on_surface.csv"): continue #make sure not already processed
    #    start = timer()
    #    print (path)
    #    create_gaze_csv(path, surf, gazefile, steerfile, rerun)
    #    compute_time = timer()-start
    #    print("CSV build took %f seconds" % compute_time)
        #print (path)
        #main(path, surf)