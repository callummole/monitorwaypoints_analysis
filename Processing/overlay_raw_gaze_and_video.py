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

import drivinglab_projection as dp

class Global_Container(object):
    pass

def hack_camera(camera):

    """hack camera to match camera used
        see camera_models.py
         cam_name='Pupil Cam1 ID2', resolution=(1920, 1080)
    
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
    camera['resolution'] = np.array([1920, 1080])

    return camera

def correlate_data(data,timestamps):
    
    """
    timestamps is taken from world_timestamps.

    also the timestamps in the recalib file is taken from pupil. 
    """   
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps] #empty array.

    frame_idx = 0
    data_index = 0 
    
    #order by timestamp
#    data = pd.DataFrame(data)
#    data = data.sort_values(by=['timestamp'],ascending=True)
#    data = data.reset_index(drop=True)
    data = sorted(data, key = lambda x: x['timestamp'])

    while True:
        try:
            #datum = data.iloc[data_index]
            datum = data[data_index]
            ts = ( timestamps[frame_idx]+timestamps[frame_idx+1] ) / 2.
     #       print ("world TS: ", datum["timestamp"], " --- World TS: ", ts)
        except IndexError:
            break

        if datum['timestamp'] <= ts:
        #    print ("matched timstamp")
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1        
        else:
            frame_idx+=1
    return data_by_frame

def load_msgpack_gaze(file):
    
    gaze_data = []
    with open(file, "rb") as fh:
        for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):

            #each row is a packed dict containing topic, norm_pos, conf, ts, recv_ts
            data = msgpack.unpackb(payload, raw=False)                        
            gaze_data.append(data)

    return gaze_data

def undistort_norm_pos(cam, gaze_data):

    """uses camera model.py to undistort gaze data"""

    data = gaze_data
    for d in data:
        
       
        norm_pos = np.array(d['norm_pos'])
       
        refs = norm_pos*cam.resolution 
       
        refs = cam.unprojectPoints(refs, normalize=False)[:,:-1] #drop the last dimension as only dealing with 2D points.

        refs += 1.0; refs /= 2.0 #rescale to keep in 0,1 for surface tracker.
        
        d['norm_pos'] = np.squeeze(refs)
        
        #print("check attribute", d['norm_pos'])
        
        
    return data

if __name__ == '__main__':

    #rootdir = "E:/Trout_rerun_gazevideos"   
    #rootdir = "C:/git_repos/sample_gaze_Trout/Trout_501_3"
    #rootdir = "C:/git_repos/sample_gaze_Trout/"
    #rootdir = "F:/Edited_Trout_rerun_Gaze_Videos/Trout_510_1"
    rootdir = "F:/Edited_Trout_rerun_Gaze_Videos/Trout_503_1"
    video_res = 1920, 1080
    camera_spec = camera_models.load_intrinsics(directory="", cam_name='Pupil Cam1 ID2', resolution=video_res)
    K = camera_spec.K
    D = camera_spec.D

    undistort = True
        
    #print ("here")
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) 
        print (path)

        video_file = 'world.mp4'
        
        marker_path = os.path.join(path,'square_marker_cache')
        if os.path.exists(marker_path):   

            persistent_cache = Persistent_Dict(os.path.join(path,'square_marker_cache'))

            path_3D = path + "/offline_data/gaze-mappings"            

            marker_cache = persistent_cache.get('marker_cache',None)

            #camera = load_object('camera')
            #camera = hack_camera(camera)            

            video = cv2.VideoCapture(os.path.join(path,video_file))
            #print ("Opened: ",video.isOpened())        

            
            gazefile = 'Default_Gaze_Mapper-d82c07cd.pldata'
            #load raw gaze file in screen normalised positions.
            raw_gaze = load_msgpack_gaze(os.path.join(path_3D, gazefile))            
            world_timestamps = np.load(os.path.join(path, "world_timestamps.npy"))      
            raw_gaze = correlate_data(raw_gaze, world_timestamps)
            #print(raw_gaze[2][1])
            
            gaze_idx = []
            vid_x = []
            vid_y  = []
            for gaze_frame in raw_gaze:
                if gaze_frame == []: continue                
                for rg in gaze_frame:
                    gaze_idx.append(rg['index'])

                    if undistort:
                        norm_pos = np.array(rg['norm_pos'])
                        refs = norm_pos*camera_spec.resolution 
                        
                        #refs = camera_spec.unprojectPoints(refs, normalize=False)[:,:-1] #drop the last dimension as only dealing with 2D points                        
                        #refs += 1.0; refs /= 2.0 #rescale to keep in 0,1 for surface tracker.
                        #rg['norm_pos'] = np.squeeze(refs)       
                        refs = np.float32(refs).reshape(-1,1,2)
                        #print("r1", refs)
                        gaze = cv2.fisheye.undistortPoints(refs,K,D, P=K).reshape(2)
                        #print("g", gaze)
                        
                        rg["norm_pos"] = (float(gaze[0]), float(gaze[1])) #not normalised 
                        

                    vid_x.append(rg['norm_pos'][0])
                    vid_y.append(rg['norm_pos'][1])                            

            gaze_np = pd.DataFrame() 
            gaze_np['gaze_idx'] = gaze_idx
            gaze_np['vid_x'] = vid_x
            gaze_np['vid_y'] = vid_y
            
           # print(gaze_np)
            
            print ("starting plotting...")
            start_idx = gaze_np['gaze_idx'].iloc[-1] - 2000

            for idx, marker_frame_ in enumerate(marker_cache):
                
                if idx < start_idx: continue
                #plot gaze
                #take gaze from gaze_csv.
                vid_x = np.mean(gaze_np.loc[gaze_np['gaze_idx']==idx, 'vid_x'])
                vid_y = np.mean(gaze_np.loc[gaze_np['gaze_idx']==idx, 'vid_y'])                
                vid_coords = np.array([vid_x, vid_y])
                #vid_coords *= video_res                

                video.set(cv2.CAP_PROP_POS_FRAMES, idx) #set to new frame for the sake of missing data.

                #plot video.
                ret, frame = video.read()

                if undistort:
                    frame = camera_spec.undistort(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                #frame = cv2.flip(frame, 0)
                #plt.plot(vid_coords[0], video_res[1] - vid_coords[1], 'mo', markersize = 5)
                plt.plot(vid_coords[0],vid_coords[1], 'mo', markersize = 5)
                #plt.imshow(cv2.flip(frame, 0), cmap='gray')
                plt.ylim(0, video_res[1])
                plt.xlim(0, video_res[0])
                plt.imshow(cv2.flip(frame,0))
                #plt.imshow(frame)
                #plt.imshow(screenshot)
                plt.pause(.016)                    
                plt.cla()
            plt.show()   
                 

                    