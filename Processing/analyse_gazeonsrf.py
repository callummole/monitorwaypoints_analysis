###script to view and almagamate eye tracking files.
import numpy as np
import sys, os
from file_methods import *
import pickle
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import math as mt 
from scipy.interpolate import interp1d
from nslr_hmm import *
from timeit import default_timer as timer

"""
Useful docs:
    
To understand marker transform:    
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

Marker centroids & vertices appear to be given in the camera's resolution.

Steps:
    1) Plot markers and gaze position per frame to explore data.
    2) Save in csv file with experiment and trial ID.
    3) Use matlab script -- converted into python -- to generate measures, such as angles etc. (or could do that in this script)
    
    #Screen co-ords.
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+
    
Now I have the markers positions I need to do more with them than simple translate them into screen co-ords.
I also need to correct for tilt of the camera, and correct gaze position in relation to this. 
Look at how their screen tracker and surface transformations do this. 
    
"""







def correlateTimestamps(df):
   
    #TODO: use the eye_id to average over duplicate timestamps with multiple eyes. 
    #FIXME: This function is slow and doesn't work properly. Dealing with dataframes in this way seems slow. 
    
    #receives panda dataframe and crunches the multiple timestamps.
    world_ts = np.unique(df['world_timestamp'])
    data_by_ts = pd.DataFrame()    
        
    
    def avgAcrossTs(stamps):
        #receives a list of entries associated with that timestamp.
        #take the most confidence. If all equal, average.
        print(stamps)
        mx = stamps['confidence'].max() 
        gp= stamps.loc[stamps['confidence']==mx]
        avggp = gp.mean(0)        
        return avggp
    
    #loop through surface_timestamps and find closest frame index 
    
    for row in range(df.shape[0]):
        #print(row)
        datum = df.iloc[[row]] #needs list selection for dataframe.   
        surf_ts = df.iloc[row,1] #surface timestamp        
        if surf_ts >0 and surf_ts<20:    #check ts is during trial. If not don't process datum.
            tsdiff = world_ts - surf_ts 
            idx = np.argmin(abs(tsdiff))
            datum['frame_index'] = idx #Add new column.
            data_by_ts = pd.concat([data_by_ts,datum])
                
    #now you've correlated timestamps. Return average for duplicate entries. 
    #
    df_new = data_by_ts.groupby(['frame_index']).apply(avgAcrossTs)
   
#    for frame in range(len(world_ts)):
#        
#        #find all the entries.
#        ts_entries = []
#        for i in range(len(data_by_ts)):
#            dt = data_by_ts[i]
#            if dt['frame_index'] == frame:
#                ts_entries.append(i)
#                
#        if len(ts_entries) > 0:
#            Tlist = [data_by_ts[ts_entries[i]] for i in range(len(ts_entries))]    
#            Tdf = pd.DataFrame(Tlist)          
#            d= avgAcrossTs(Tdf)
#            df_new.append(d)            
    
    return df_new
    
def GazeAngles(df):
    
        
    def SurfaceToGazeAngle(gp):
        
        #proj_width = 1.965
        #proj_height = 1.115    
        
        #pixel size of markers of total white border marker is 118 x 107. But I think surface is up to black marker edge.
        #measurements of black marker edge in inkscape are ~75 x 72 pixels. 
        #NEED REAL_WORLD MEASUREMENTS OF SURFACE SIZE UP TO BLACK SQUARE.
        #AND HORIZON RELATIVE TO BOTTOM AND TOP OF SQUARE.
        #put the below into a function and call apply: 

        #need to check that surfaces is from edges, as in vizard the placement is the bottom left corner
        w = 1.65 #real-world size of surface, in m.
        h = .603
        
        centrex = .5 #horiz centre of surface is centre of experiment display.
        centrey = .74 #placement of horizon in normalised surface units. Minus off norm_y to get gaze relative to horizon.
        #TODO: CHECK HORIZON MEASUREMENT AS SURFACE DOESN'T ALIGN WITH TOP OF MARKERS. NEED TO CORRECT FOR DISTORTION.
        screen_dist = 1.0 #in metres
        
        #convert the scale to real-distances from centre.
        x = gp['x_norm']
        y = gp['y_norm']
        real_h = (x-centrex)*w
        real_v = (y-centrey)*h
#	
#	
    	#calculate gaze angle
        hrad = mt.atan(real_h/screen_dist)
        vrad = mt.atan(real_v/screen_dist)
#	
    #	#convert to degrees
        hang = (hrad*180)/mt.pi
        vang= (vrad*180)/mt.pi
#	
        return (hang, vang) 
    
    
    df['hangle'], df['vangle'] = zip(*df.apply(SurfaceToGazeAngle,axis=1))
       
    return df	
    
def StitchSteering(df, pcode, pid, ecode,steerdir, trialn):
    #passed the code, id, and ecode, steerdir and trialn. Use this to pick the correct file.
    filename = steerdir + "/" + pcode + "/" + ecode + '_' + pcode + '_' + trialn + '.dat' #formula for reading in steering file. 
    steerdata = pd.read_table(filename) #read steering data
    

#def GazeinWorld(df):
    
    #convert matlab script to python to calculate gaze from centre of the road, using Gaze Angles.

if __name__ == '__main__':
	#rootdir = sys.argv[1] 
    #rootdir = "E:/Masters_17-18_RunningTotal_Lastupdate_250118/PG_AW15F" #directory for eyetracking files. 
    #rootdir = "E:/Masters_17-18_RunningTotal_Lastupdate_250118/KH12F/Crow17_KH12F/001"
   # rootdir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Master_SampleParticipants/"
    #rootdir = "F:/MastersEyeTracking_140218/"
    #rootdir = "E:/EyeTrike_Backup/Recordings/Masters_17-18_RunningTotal_Lastupdate_010318/"
    #rootdir = "E:/EyeTrike_Backup/Recordings/Masters_17-18_RunningTotal_Lastupdate_130418/"   
    rootdir = "D:/EyeTrike_Backup/Recordings/Masters_17-18_RunningTotal_Lastupdate_130418/"    
    #rootdir = "E:/EyeTrike_Backup\Recordings/Masters_17-18_RunningTotal_Lastupdate_130418/HP18F_2/Crow17_HP18F"
    #save the pupil-corrected files in the analysis folder.
    #savedir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Data/EyetrackingData/"
    savedir = "M:/EyetrackingData/"
    resave = False #boolean whether to move files to savedir
    steerdir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Master_SampleParticipants/SteeringData/"
    resx, resy = 1280,720
   # marker_corners_norm = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
   # marker_box = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32) #this is for perspectiveTransform.
    
    print (rootdir)
    
    nC = 4 #number of conditions.
    Trialidx = np.zeros(nC)
    
    #varnames = np.array('pp','tn','cn','Fix','Bend','Exp',')
    #measures: LookaheadDistance; HighestFixationDensity,  
    #create a dataframe for averaged measures 
    #MATLAB outtable = cell2table(out,'VariableNames',{'pp', 'tn', 'cn', 'Fix', 'Bend', 'SB','RMS',...
    #'AvgYaw','StdYaw','AvgSWA','StdSWA','AvgSWVel'});
    
    #pcodes = ['AW15F','LH28M']
    #pcodes = ['KH12F']
#    pcodes = ['LB04F','MS05M','AT13F','AW15F','SW08F','JO03M','SC20F','LH28M',
            #  'HC15F','HR23F','BM29F','LW14F','KH12F']
    pcodes = ['LB04F','MS05M','AT13F','AW15F','SW08F','JO03M','SC20F','LH28M',
              'HC15F','KH12F','HR23F','BM29F','LW14F','LW04F','JH12M','OS08F','EI07F',
              'EL17F','BN31F','CS14F','EL06F','HP18F_2','LH22F','OH20M'] #need the HP18F_2 due to corrupted folder 'HP18F_2'
    #pcodes = ['HP18F_2']
    expcodes = ['NoLoad1','Crow17','NoLoad2','Sparrow17']
    cndtcodes = ['Free_Free','Free_Fix','Fix_Free','Fix_Fix']
   #expcodes = ['Crow17']
        
    master_df = pd.DataFrame() #master data for gaze file.
    master_segment_df = pd.DataFrame() #master data for segment file.                
    
    for pp in pcodes:
        for exp in expcodes:
            pfolder = rootdir + pp + '/' + exp + '_' + pp #formula for folder naming.
            print(pfolder)
            #print ("here")
            for dirs in os.walk(pfolder):
                path = str(dirs[0]) 
                print (path)
                
                if os.path.exists(path + '/gaze_on_surface_Corrected_includingID.csv'):
                    
                    #start = timer()
#                   
                    trial = path[-3:] #trial number is the three digits before 
                    if trial == '000':
                        trial = 0
                    else:
                        trial = trial.lstrip("0") #remove leading zeros.
                    
                    #trial = int(trial) + 1 #remove zero indexing.
                        
                    print ("trial: ", trial)
                    """
                    file has columns: 'world_timestamp', 'surface_timestamp', 'x_norm', 'y_norm', 'on_srf', 'confidence'
                    There may be more than one estimate for each timestamp, and they may not be in order.
                    Where they is more than one, either: Take the higher conf, or take the avg if conf is equal.
                    
                    STEPS:
                    1) Function to convert norm_pos to x&y angular deviation from centre of screen (horizon). DONE
                    2) Function to locate angles in-world: Lookahead distance; Gaze relative to centre of road.
                    3) Output as csv in trial root folder.
                    4) Function to calculate some basic measures dispersal measures from angular deviation. These are average measures per trial
                    5) Measure up with condition flags from the annotation at the start of the trial.
                    5) Collate measures across participants into long-format for R processing. 
                    
                    """                                        
                       
                    df = pd.read_csv(path+'/gaze_on_surface_Corrected_includingID.csv', sep=',',header=0)
                    #df = pd.read_csv(path+'/gaze_on_surface_Corrected.csv', sep=',',header=0)
                    
                    #no need to do return one estimate per world timestamp (i.e. frame index). Can instead use the gaze timestamp, which is monotonically increasing. 
                    #df = correlateTimestamps(df) #returns one estimate per world timestamp
                    df['ppcode'] = pp #store participant code.
                    df['pp_id'] = pcodes.index(pp)+1 #store ppnumber
                    df['exp_id'] = exp
                    df['trialn'] = trial
                    
                    #put trial type into longformat, parsing out bend and condition.
                    ci = df.loc[1,'trialtype']
                    if ci < 0:
                        Bend = 'Left'
                    else:
                        Bend = 'Right'
                    
                    print ('trialtype: ', ci)
                    
                    condition = cndtcodes[abs(ci)-1]
                    df['Bend'] = Bend
                    df['condition'] = condition
                    
                    
                    ####check interpolate functions####                    
#                    yinterpolater = interp1d(df['gaze_timestamp'],df['y_norm'])
#                    y1 = yinterpolater(df['world_timestamp'])
#                    
#                    xinterpolater = interp1d(df['gaze_timestamp'],df['x_norm'])
#                    x1 = xinterpolater(df['world_timestamp'])
#                                    
#                    
#                    plt.plot(x1,y1,'ro',df['x_norm'],df['y_norm'],'bo')
                    
#                    plt.show()                    
                    
                    df = GazeAngles(df) #adds two new columns: hangle and vangle 
                  #  df = StitchSteering(df, pp,pcodes.index(pp),exp,steerdir,trial)
                    #This is where I need to load the steering data in and stitch it together.
                    #Can just load it into a list. 
                                        
                    #    df = GazeinWorld(df) #locates gaze in world. 
                    
                    ###here we add the segmentation, using jami's algorithm.                    
                    #Need to use gaze_timestamp as it is monotonically increase, whereas world_timestamp is linked to the frame video so has multiple recordings per frame.
                    v = df['vangle'].values
                    h= df['hangle'].values
                    t = df['gaze_timestamp'].values
                    eye = np.vstack((v,h)).T
                    print('classifying gaze')
                    sample_class, segmentation, seg_class = classify_gaze(t, eye)
                    #sample_class is an array of same dimensions as inputs to classify_gaze, so can be added on to original dataframe.
                    df['sample_class'] = sample_class
                    
                    
                    #seg_class is an array of dimensions the same as number of segmentations
                    #segmentation is an nslr.slow class, with segments that have t and x. 
                    #t is a 2dim array with start and end points. x is a 2x2 array vangle in x[:,0] and hangle in [x:,1]
                    #plt.plot(t,h,'.')
                    
                    #need to save segment in own file as it has different dimensions.
                    seg_trial = pd.DataFrame()
                    #add segmentation class and identification variables                    
                    seg_trial['seg_class'] = seg_class
                    seg_trial['ppcode'] = pp #store participant code.
                    seg_trial['pp_id'] = pcodes.index(pp)+1
                    seg_trial['exp_id'] = exp
                    seg_trial['trialn'] = trial
                    seg_trial['condition'] = condition
                    seg_trial['Bend'] = Bend                    
                    
                    for i, segment in enumerate(segmentation.segments):                                                
                        t = np.array(segment.t) # Start and end times of the segment
                        x = np.array(segment.x) # Start and end points of the segment
                        seg_trial.loc[i,'t1'] = t[0]
                        seg_trial.loc[i,'t2'] = t[1]
                        seg_trial.loc[i,'v1'] = x[0,0]
                        seg_trial.loc[i,'v2'] = x[1,0]
                        seg_trial.loc[i,'h1'] = x[0,1]
                        seg_trial.loc[i,'h2'] = x[1,1]
                      #  plt.plot(t, x[:,1], 'ro-', alpha=0.5)
                          
                    #plt.show()
                    master_segment_df = pd.concat([master_segment_df,seg_trial])
                    

            
               #print ("added to master df")
                    master_df = pd.concat([master_df,df])
                    
                    #compute_time = timer()-start
                    #print("classifying gaze took %f seconds" % compute_time)
                
    
    #now you've built the master data of all trials, save it.
    master_df.to_csv(savedir + "FramebyFrameData_longFormat_180418.csv")        
    master_segment_df.to_csv(savedir + "SegmentationData_longFormat_180418.csv")



