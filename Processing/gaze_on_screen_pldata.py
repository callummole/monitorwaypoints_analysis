#!/usr/bin/env python3

##script corrects for camera distortion and extracts the marker's position. Saves "pupil-corrected" in each data folder.
#original script at samtuhka/gazesim_tools

import sys, os
import cv2
#import argh
import pickle
import square_marker_detect as markerdetect
import numpy as np
from datetime import timedelta
from shutil import copyfile
from file_methods import *
from gaze_to_world_coordinates import undistort_points_fisheye
from timeit import default_timer as timer

import camera_models

def denormalize(pos, width, height, flip_y=False):
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1-y
    y *= height
    return x,y

def normalize(pos, width, height, flip_y=False):
    """
    normalize return as float
    """
    x = pos[0]
    y = pos[1]
    x /=float(width)
    y /=float(height)
    if flip_y:
        return x,1-y
    return x,y


def correct_gaze_positions(path, K, D, cm, res):

    print ("correcting gaze positions")
    gaze_pl = load_pldata_file(path, 'gaze') #tuple containing data, timestamps, and topics.
    gazes = gaze_pl[0]
    with PLData_Writer(path, "gaze_corrected") as output:
        for g in gazes:
            g = dict(g)
            gaze = denormalize(g['norm_pos'],res[0], res[1])
            gaze = np.float32(gaze).reshape(-1,1,2)
            gaze = cv2.fisheye.undistortPoints(gaze,K,D, P=cm).reshape(2)
            gaze = normalize(gaze,res[0], res[1])
            g["norm_pos"] = (float(gaze[0]), float(gaze[1]))
            output.append(g)


if __name__ == '__main__':
    #argh.dispatch_command(marker_positions)
    #rootdir = sys.argv[1] #picks up directory folder given as argument. First argument is root directory, second argument is a boolean for rewrite or not. Default is not.
    
    rootdir = "F:\\EyeTrike_Backup\\Recordings\\Trout_18_rerun"

    os.chdir(sys.path[0]) # change directory to root path.
       
    for dirs in os.walk(rootdir): #does it for all dirs in that folder
        path = str(dirs[0]) + "/"
        
        #correcting gaze without the video processing.

        if not os.path.exists(path + "gaze_corrected.pldata"):
                        
            start = timer()
                
            print ("resaving gaze")
                
            ####### get correct camera model ##########

            mymodel = camera_models.load_intrinsics(directory="", cam_name='Pupil Cam1 ID2', resolution='(1920, 1080)')
            ############################################
            #seems like two camera matrixes are the same?
            K = mymodel.K
            D = mymodel.D
            cm = mymodel.K #rect_camera_matrix in other code, but set to camera_spec.K
            res = mymodel.resolution
            correct_gaze_positions(path, K, D, cm, res):
                
            compute_time = timer() - start
                
            print("correcting gaze took %f seconds" % compute_time)

	#here also should include save annotations. 
