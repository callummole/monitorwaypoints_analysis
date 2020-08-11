import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from nslr_hmm import *
from timeit import default_timer as timer
from pprint import pprint

def my_initial_probabilities():

    transition_model = my_gaze_transition_model()   
    initial_probabilities = np.ones(len(transition_model))
    initial_probabilities[2] = 0 #PSO is index three.
    initial_probabilities /= np.sum(initial_probabilities)

    return initial_probabilities


def my_gaze_transition_model():
    transitions = np.ones((4, 4))
    #transitions[0, 2] = 0
    transitions[:,2] = 0 #zero the PSO probability. 
    transitions[2, 1] = 0
    #transitions[3, 2] = 0
    transitions[3, 0] = 0.5
    transitions[0, 3] = 0.5
    
    for i in range(len(transitions)):
        transitions[i] /= np.sum(transitions[i])

    return transitions

def classify_gaze_adjusted(ts, xs, **kwargs):
    fit_params = {k: kwargs[k]
            for k in ('structural_error', 'optimize_noise', 'split_likelihood') if k in kwargs
    }
    segmentation = nslr.fit_gaze(ts, xs, **fit_params)
    seg_classes = classify_segments(segmentation.segments, transition_model=my_gaze_transition_model(), initial_probabilities=my_initial_probabilities())
    sample_classes = np.zeros(len(ts))
    for c, s in zip(seg_classes, segmentation.segments):
        start = s.i[0]
        end = s.i[1]
        sample_classes[start:end] = c

    return sample_classes, segmentation, seg_classes

def main(datafilepath, outfilepath):
    
                                                                   
    steergaze_df = pd.read_csv(datafilepath, sep=',',header=0)  
    steergaze_df = steergaze_df.query("dataset == 2").copy()              
    new_gaze_file = os.path.splitext(datafilepath)[0] + '_addsample.csv'
    #steergaze_df = pd.read_feather(datafilepath)                

    master_segment = pd.DataFrame()    
    master_steergaze = pd.DataFrame()

    #for trial in picked_trials:
    for trialcode, trialdata in steergaze_df.groupby('trialcode'):

        begin = timer()
        
        print("Classifying gaze: ", trialcode)                    
                        
        trialdata = trialdata.copy()

        #print(list(trialdata.columns))
        
        quality_gaze = trialdata.loc[trialdata['confidence']>.6,:].copy()

        h = quality_gaze['hangle'].values
        v = quality_gaze['vangle'].values
        t = quality_gaze['currtime'].values
                
        eye = np.vstack((v,h)).T                
        #sample_class, segmentation, seg_class = classify_gaze(t, eye)
        print("classification without PSO")
        scale_factor = 3
        try:
            if eye.shape[0] < 20:
                raise Exception("less than 20 gaze data points in trial")
            sample_class, segmentation, seg_class = classify_gaze_adjusted(t, eye*scale_factor)
        except Exception as e:
            print("Cannot process trial" + str(trialcode), e) 
            quality_gaze['sample_class'] = np.nan                  
            cols_to_use = list(quality_gaze.columns[:-1]) #all bu sample_class
            #print(cols_to_use)
            trialdata = pd.merge(trialdata, quality_gaze, on = cols_to_use, how='outer')    
            master_steergaze = pd.concat([master_steergaze, trialdata])
            continue

        #sample_class is an array of same dimensions as inputs to classify_gaze, so can be added on to original dataframe.
        quality_gaze['sample_class'] = sample_class                   
        #pprint(quality_gaze)
        #pprint(trialdata)
        #add back into trialcode, with NaN for samples that were below the threshold
        cols_to_use = list(quality_gaze.columns[:-1]) #all bu sample_class
        
        trialdata = pd.merge(trialdata, quality_gaze, on = cols_to_use, how='outer')
        #pprint(trialdata)        
        #seg_class is an array of dimensions the same as number of segmentations
        #segmentation is an nslr.slow class, with segments that have t and x. 
        #t is a 2dim array with start and end points. x is a 2x2 array vangle in x[:,0] and hangle in [x:,1]
        #plt.plot(t,h,'.')
        cols = ['seg_class','ID','trialcode','condition','count','drivingmode','sectionorder','block',
        't1','t2','v1','v2','h1','h2','yawrate','roadsection','th_1','th_2','on_srf_1','on_srf_2','T0_1','T0_2','seg_i'] #'gtp_1','gtp_2','seg_i']
        seg_trial = pd.DataFrame(columns=cols)
        block, ppid, sectiontype, condition, count =  trialcode.split("_") 

        seg_trial['seg_class'] = seg_class                        
        seg_trial['ID'] = ppid
        seg_trial['trialcode'] = trialcode
        seg_trial['condition'] = condition
        seg_trial['count'] = count
        seg_trial['drivingmode'] = sectiontype
        seg_trial['sectionorder'] = trialdata['sectionorder'].values[0]
        seg_trial['block'] = block       


        for i, segment in enumerate(segmentation.segments):                                                
            t = np.array(segment.t) # Start and end times of the segment
            x = np.array(segment.x) # Start and end points of the segment
            x /= scale_factor
            seg_trial.loc[i,['t1', 't2']] = t            
            #seg_trial.loc[i,'t2'] = t[1]
            seg_trial.loc[i,'v1'] = x[0,0]
            seg_trial.loc[i,'v2'] = x[1,0] 
            seg_trial.loc[i,'h1'] = x[0,1]
            seg_trial.loc[i,'h2'] = x[1,1]
                                
            #here calculate average yaw-rate for segment for the corresponding time period.
            
            start = trialdata['currtime'] >= t[0]
            end = trialdata['currtime'] <= t[1] 

            yawrates = trialdata.loc[start&end,'yawrate'] # degrees per second?
            seg_trial.loc[i,'yawrate'] = yawrates.mean()            


            #here add all the other variables you want, e.g. time headway. 
            launch_row = trialdata.loc[trialdata['currtime']==t[0],:]
            land_row = trialdata.loc[trialdata['currtime']==t[1],:]        

               
            
            #there should be two entries for every currtime. the launch one is the last one, the land is the first one.
            seg_trial.loc[i,'drivingmode'] = launch_row['drivingmode'].values[0]
            seg_trial.loc[i,'roadsection'] = launch_row['roadsection'].values[0]
            seg_trial.loc[i,'th_1'] = launch_row['th_along_midline'].values[0]
            seg_trial.loc[i,'th_2'] = land_row['th_along_midline'].values[0]
            seg_trial.loc[i,'on_srf_1'] = launch_row['on_srf'].values[0]
            seg_trial.loc[i,'on_srf_2'] = land_row['on_srf'].values[0]
            seg_trial.loc[i,'T0_1'] = launch_row['T0_on_screen'].values[0]
            seg_trial.loc[i,'T0_2'] = land_row['T0_on_screen'].values[0]
            #seg_trial.loc[i,'gtp_1'] = launch_row['gazetopath_angulardistance'].values[0]    
            #seg_trial.loc[i,'gtp_2'] = land_row['gazetopath_angulardistance'].values[0]    
            seg_trial.loc[i, 'seg_i'] = i

        #master_segment = pd.concat([master_segment,seg_trial])
        #master_steergaze = pd.concat([master_steergaze, trialdata])

        compute_time = timer()-begin
        print("Processing trial took %f seconds" % compute_time)

        print("APPENDING SEG")
        with open(outfilepath, 'a', newline = '') as segfile:
            seg_trial.to_csv(segfile, mode='a', header=segfile.tell()==0)


        print("APPENDING STEERGAZE")
        with open(new_gaze_file, 'a', newline = '') as steergazefile:
            trialdata.to_csv(steergazefile, mode='a', header=steergazefile.tell()==0)
        

if __name__ == '__main__':

    #datafilepath = "E:/Trout_rerun_steering_pooled"
    outfile = "E:/segmentation_trout_rerun_160919.csv"

    datafilepath = "../Post-processing/trout_rerun.feather"
    main(datafilepath, outfile)


