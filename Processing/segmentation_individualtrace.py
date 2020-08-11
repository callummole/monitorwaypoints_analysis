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

def main(datafilepath, outfilepathm, trialcode):
    

    steergaze = pd.read_csv(datafilepath, sep=',',header=0)        
    #query_string = "trialcode == " + trialcode
    query_string = f"trialcode == '{trialcode}'"
    
    trialdata = steergaze.query(query_string)    
    trialdata.sort_values('currtimezero', inplace=True)        

    trialdata = trialdata.loc[trialdata['confidence']>.6,:]    

    h = trialdata['hangle'].values
    v = trialdata['vangle'].values
    t = trialdata['currtimezero'].values
                
    eye = np.vstack((v,h)).T                    

    tmodel = my_gaze_transition_model()
    print("tmodel")
    pprint(tmodel)
    pprint(eye)


    #oldmodel = gaze_transition_model()
    #pprint(oldmodel)

    #initprobs = my_initial_probabilities()
    #print("initprob")
    #pprint(initprobs)
    print("default classification")
    sample_class, segmentation, seg_class = classify_gaze(t, eye)
        #sample_class is an array of same dimensions as inputs to classify_gaze, so can be added on to original dataframe.
            
    scale_factor = 1.8
    print("classification without PSO")
    new_sample_class, new_segmentation, new_seg_class = classify_gaze_adjusted(t, eye*scale_factor)
    


    #PLOT
    class_color= {
        1: 'r',
        2: 'b',
        3: 'g',
        4: 'y',
        5: 'm',
        6: 'c',
        22: 'orange'
    }
    
    plt.figure(1)  		
    ax1 = plt.subplot(2,1,1)
    
    for i, segment in enumerate(segmentation.segments):                                                
            t = np.array(segment.t) # Start and end times of the segment
            x = np.array(segment.x) # Start and end points of the segment    
            v = x[:,0]                        
            h = x[:,1]
            plt.plot(t, v, '-', color=class_color[seg_class[i]])			
            #plt.xlim(5,15)			
        
    plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
    for i, segment in enumerate(new_segmentation.segments):                                                
            t = np.array(segment.t) # Start and end times of the segment
            x = np.array(segment.x) # Start and end points of the segment    
            x /= scale_factor
            v = x[:,0]
            h = x[:,1]
            
            plt.plot(t, v, '-', color=class_color[new_seg_class[i]])						
            #plt.xlim(5,15)		

    plt.show()


    
if __name__ == '__main__':

    datafilepath = "../Data/trout_gazeandsteering_161019.csv"
    trialcode = "2_502_5_2_0"    
    outfile = ""
    main(datafilepath, outfile, trialcode)


