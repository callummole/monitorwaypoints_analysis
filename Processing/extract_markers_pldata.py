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
import calhelper as ch

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

def load_pickled_camera(file):

    camera = pickle.load(open(file, 'rb'), encoding='bytes')
    image_resolution = camera[b'resolution']
    
    if b'rect_map' not in camera:
        camera_matrix = camera[b'camera_matrix']
        camera_distortion = camera[b'dist_coefs']
        rect_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, image_resolution, 0.0)
        rmap = cv2.initUndistortRectifyMap(
            camera_matrix, camera_distortion, None, rect_camera_matrix, image_resolution,
            cv2.CV_32FC1)
    else:
        rmap = camera[b'rect_map']
        rect_camera_matrix = camera[b'rect_camera_matrix']


    K, D, resolution, cm = camera[b'camera_matrix'], camera[b'dist_coefs'], camera[b'resolution'], rect_camera_matrix


    camera = {}
    camera['camera_matrix'] = rect_camera_matrix
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
    camera['resolution'] = image_resolution
    
    return(camera, rmap)

def marker_positions(camera_spec, videofile, outfile, path, new_camera=None, start_time=0.0, end_time=float("inf"), visualize=False,
        output_camera=None, correct_gaze = False, rewrite = False, rerun = True):
    #camera = pickle.load(open(camera_spec, 'rb'), encoding='bytes')


    #image_resolution = camera[b'resolution']
    if rerun:
        camera = {}
        camera['camera_matrix'] = camera_spec.K
        camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
        camera['resolution'] = camera_spec.resolution
        resolution = 1920, 1080
    else:
        pickled_file = 'sdcalib.rmap.full.camera_unix.pickle'
        camera, rmap = load_pickled_camera(pickled_file)
        resolution = 1280, 720
    
    #print ("Videofile: ", videofile)
    video = cv2.VideoCapture(videofile)
    #print ("Opened: ",video.isOpened())
    video.set(cv2.CAP_PROP_POS_MSEC, 0.0)#start_time)#*1000)
    #print ("Video Frame Count:",video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    #marker_tracker = markerdetect.MarkerTracker()
    
    prev_minute = 0.0
    marker_cache = []

    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if rewrite:
        print ('re-writing')
        out = cv2.VideoWriter(path + "world.mp4",fourcc, 30.0, (resolution[0],resolution[1]))

    print ("starting vid...")
    markers = None
    while True:

        ret, oframe = video.read()
 #       print ("Ret: ", ret)
 #       print ("oframe:", oframe)
        if not ret:
            print ("reached")
            break
        #gaze = np.array([[500,500]])
        #cv2.circle(oframe, (int(gaze[0][0]), int(gaze[0][1])), 10, (255,255,255), thickness = -1)
        #cv2.imshow('orig', oframe)

        
        if rerun:
            frame = camera_spec.undistort(oframe)
        else:
            frame = cv2.remap(oframe, rmap[0], rmap[1], cv2.INTER_LINEAR)

        #cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 1, (0,0,0), thickness = -1)



        #print(oframe[0], oframe.shape)
        #cv2.imshow('rect', frame)
        #print(gaze)
        #cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 5, (0,255,0), thickness = -1)
        #cv2.imshow('alt_rect', frame)
        #gaze = cv2.remap(gaze.T, rmap[0], rmap[1], cv2.INTER_LINEAR)
        #print(gaze)
        #cv2.circle(frame, (int(gaze[0]), int(gaze[1])), 5, (0,255,0), thickness = -1)
        #cv2.imshow('alt_rect', frame)
        #print(cv2.remap(np.array([[5,6], [1,7]]), rmap[0], rmap[1], cv2.INTER_LINEAR))
        if rewrite:            
            out.write(frame)
        #cv2.imshow('frame', frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        msecs = video.get(cv2.CAP_PROP_POS_MSEC)
        time = msecs/1000.0
        print ("time: ", round(time,1), "s")
        if time > end_time:
            break

        #if time - prev_minute > 60.0:
        #    print >>sys.stderr, timedelta(seconds=time)
        #    prev_minute = time
        markers = markerdetect.detect_markers_robust(frame, 5, markers)
        #markers = markerdetect.detect_markers(frame, 5, min_marker_perimeter=20) #changed marker size.
       #print ("Markers: ", markers) #check code is finding markers
        #print(markers)
        marker_cache.append(markers)
        #markers = marker_tracker.track_in_frame(frame, 5)
        frame_d = {
                'ts': time,
                'markers': markers,
                }
        frames.append(frame_d)
        
        if not visualize: continue
        markerdetect.draw_markers(frame, frame_d['markers'])
        #for marker in markers:
        #    for i, corner in enumerate(marker['verts']):
        #        cv2.putText(frame, str(i), tuple(np.int0(corner[0,:])),
        #                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,100,50))
        cv2.imshow('frameBG', frame)
        cv2.waitKey(1)
    np.save(outfile, frames)
   # print(rewrite)

    if rewrite:
        print ("attempting to save:", out.isOpened())  
        out.release()

    if correct_gaze:
        correct_gaze_positions(path, K, D, cm, resolution)

    return marker_cache

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


def main(gazefolderdir, rerun = True):
    #argh.dispatch_command(marker_positions)
    #rootdir = sys.argv[1] #picks up directory folder given as argument. First argument is root directory, second argument is a boolean for rewrite or not. Default is not.
    #rootdir = "F:\\EyeTrike_Backup\\Recordings\\Trout_18_rerun"    

    rootdir = gazefolderdir
    os.chdir(sys.path[0]) # change directory to root path.
       
    for dirs in os.walk(rootdir): #does it for all dirs in that folder
        path = str(dirs[0]) + "/"
        
        folder_skip = "2B_reviewed"
        if folder_skip in path: continue
        if os.path.exists(path + "world.mp4"): # and not os.path.exists(path + 'markers.npy'):
            #rewrite = sys.argv[2]
#            rewrite = True #whether to rewrite the original file..
            rewrite = False #whether to rewrite the original file..
            print(path)
            #if os.path.exists(path + "world_old.mp4"):
            #    continue
            vid_path = path  + 'world.mp4'
            if rewrite and not os.path.exists(path + "world_old.mp4"):
                os.rename(vid_path, path + "world_old.mp4")
                vid_path = path + "world_old.mp4"
            else:
                rewrite = False                                    
                
            if os.path.exists(path + 'square_marker_cache'):          
                filename = ch.check_exist(path + 'square_marker_cache_old')  
                os.rename(path + 'square_marker_cache', filename)
                    
                
            start = timer()
            
            print ("extracting")
            marker_cache = Persistent_Dict(path + 'square_marker_cache')
            marker_cache['version'] = 2
#                markers = marker_positions('sdcalib.rmap.full.camera.pickle', vid_path, path + 'markers.npy', path, visualize = False, correct_gaze = True, rewrite = rewrite)

            #switch calib to pupil provided.

            ####### get correct camera model ##########
            
            mymodel = camera_models.load_intrinsics(directory="", cam_name='Pupil Cam1 ID2', resolution=(1920, 1080))
            

            ############################################
            
            #make sure correct gaze is false
            markers = marker_positions(mymodel, vid_path, path + 'markers.npy', path, visualize = False, correct_gaze = False, rewrite = rewrite, rerun = rerun)
            marker_cache['marker_cache'] = markers
            marker_cache['inverted_markers'] = False
            marker_cache.close()
            
            compute_time = timer() - start
        
            print("extracting markers took %f seconds" % compute_time)

	#here also should include save annotations. 

if __name__ == '__main__':

    gazefolderdir = "E:/Trout_rerun"
    main(gazefolderdir)