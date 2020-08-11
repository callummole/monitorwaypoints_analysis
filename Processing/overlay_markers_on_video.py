import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from file_methods import *
import cv2
import camera_models
from offline_reference_surface import Offline_Reference_Surface
from offline_surface_tracker import Offline_Surface_Tracker
from pprint import pprint
import square_marker_detect
from datetime import datetime
import pickle

class Global_Container(object):
    pass

def hack_camera(camera, res):

    """hack camera to match camera used
        see camera_models.py
         cam_name='Pupil Cam1 ID2', resolution=(1920, 1080)

         #for original:
        cam_name='Pupil Cam1 ID2', resolution=(1280, 720)
    
    """
    #should the dist_coefs be zero or matched to the pre-recorded calibrations?
    #since the square_marker_cache is already undistorted you do not want pass any distortion coefficients.
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])

    #camera['dist_coefs'] = np.array([
    #            [-0.1804359422372346],
    #            [0.042312699050507684],
    #            [-0.048304496525298606],
    #            [0.022210236517363622]])
    camera['camera_matrix'] = np.array([
        [843.364676204713, 0.0, 983.8920955744197],
                [0.0, 819.1042187528645, 537.1633514857654],
                [0.0, 0.0, 1.0]])


    camera['resolution'] = np.array(res)

    return camera

def load_pickled_camera(file):

    camera = pickle.load(open(file, 'rb'), encoding='bytes')
    image_resolution = camera[b'resolution']
    
    cmat = camera[b'camera_matrix']
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


if __name__ == '__main__':

    #rootdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/Trout_20_1"   

    rerun = False
    
    
    
    if rerun: 
        rootdir = "D:/Trout_rerun_gazevideos"
    else:
        rootdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/Trout_10_1"

    #rootdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/Trout_3_2"
    marker_filename = 'square_marker_cache'


    #old_file = 'sdcalib.rmap.full.camera_unix.pickle'
    #c, rmap, camera_mat = load_pickled_camera(old_file)
    
    if rerun:
        video_res = 1920, 1080
        camera_spec = camera_models.load_intrinsics(directory="", cam_name='Pupil Cam1 ID2', resolution=video_res)
        camera = {}
        camera['camera_matrix'] = camera_spec.K
        camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
        camera['resolution'] = video_res

    else:
        video_res = 1280, 720
        old_file = 'sdcalib.rmap.full.camera_unix.pickle'
        camera, rmap = load_pickled_camera(old_file)
    #print(pfolder)
    
    #print ("here")
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) 
        print (path)

        video_file = 'world.mp4'
        
        marker_path = os.path.join(path,marker_filename)
        if os.path.exists(marker_path):   

            persistent_cache = Persistent_Dict(os.path.join(path,'square_marker_cache'))
            

            """TODO: 
            To save time only start plotting once the idx that the gaze_on_surface file starts with is reached.
        


            """
            #date = datetime.date(datetime.now())
            #filename = 'gaze_on_surface_' + str(date) + '.csv'
            #gazedata = pd.read_csv(os.path.join(path, filename))

            
            #start_idx = gazedata['world_frame_idx'].values[0]
            #print("start:", start_idx)
            if rerun: 
                start_idx = 4100
            else:
                start_idx = 30

            marker_cache = persistent_cache.get('marker_cache',None)

            #camera = load_object('camera')
            #camera = hack_camera(camera)

            

            video = cv2.VideoCapture(os.path.join(path,video_file))
            #print ("Opened: ",video.isOpened())        
            video.set(cv2.CAP_PROP_POS_FRAMES, start_idx) #set to beginning frames.

            print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print ("starting plotting...")

            """
            TODO:

            use reference_surface.gl_draw_frame.py for the perspective transform code.


            """

            ###make surface
            surface_definitions = Persistent_Dict(os.path.join(path,'surface_definitions'))

            #pprint(surface_definitions)

            marker_definitions = surface_definitions.get('realtime_square_marker_surfaces')[0]
            g_pool = Global_Container()
            g_pool.rec_dir = path
            s = Offline_Reference_Surface(g_pool, saved_definition=marker_definitions)
            
            #retrieve video.
            marker_frame = None
            #prev_frame = None

            for idx, marker_frame_ in enumerate(marker_cache):                
                if idx < start_idx: continue

                #plot video.
                ret, oframe = video.read()        
                
                assert((oframe.shape[1],oframe.shape[0]) == video_res)
                #frame = oframe
                
                            
                if rerun:
                    frame = camera_spec.undistort(oframe)
                else:
                    
                    frame = cv2.remap(oframe, rmap[0], rmap[1], cv2.INTER_LINEAR)
                    
                
                #frame = oframe    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                
                
                # Recalculate the markers online for testing
                marker_frame = square_marker_detect.detect_markers_robust(frame, 5, marker_frame)

                
                #create surface
                s.locate(marker_frame, camera, 0, 0.0)

                surface_outline = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32)
                #hat = np.array([[[.3,.7],[.7,.7],[.5,.9],[.3,.7]]],dtype=np.float32)
                #hat = cv2.perspectiveTransform(hat,self.m_to_screen)              
                #print(s.m_to_screen)
                #try:  
                surface_outline = np.squeeze(cv2.perspectiveTransform(surface_outline,s.m_to_screen))
                #except: continue

                surface_outline *= video_res                

                plt.plot(surface_outline[:,0],surface_outline[:,1], 'g-')
                
                #plot gaze
                
                
                #plot markers
                for marker in marker_frame:
                    
                    verts = marker.get('verts')

                    for vertices in verts:

                        vertex = vertices[0]
                        plt.plot(vertex[0], video_res[1] - vertex[1], 'ro', markersize = 2)
                        plt.xlim(0, video_res[0])
                        plt.ylim(0, video_res[1])
                
                plt.imshow(cv2.flip(frame, 0), cmap='gray')
                plt.pause(.2)                    
                plt.cla()
            plt.show()        

                    
