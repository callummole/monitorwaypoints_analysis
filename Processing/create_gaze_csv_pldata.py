import os, sys
import numpy as np
from file_methods import *
from offline_reference_surface import Offline_Reference_Surface
from offline_surface_tracker import Offline_Surface_Tracker
import csv
import logging
import pandas as pd
from shutil import copyfile
from timeit import default_timer as timer
import msgpack
from datetime import datetime

import cv2

import camera_models

import matplotlib.pyplot as plt

def correlate_data(data,timestamps):
    
    """
    timestamps is array from world_timestamps. The frame indexing is the most important thing

    For D1 The 'timestamp' column in data is the pupil timestamps from pupil labs.
    For D2 the timestamp in the recalib file is pupil timestamps as received by viz.
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
            #print ("world TS: ", datum["timestamp"], " --- World TS: ", ts)
        except IndexError:
            break

        if datum['timestamp'] <= ts:
            #print ("matched timstamp")
            datum['index'] = frame_idx
            data_by_frame[frame_idx].append(datum)
            data_index +=1        
        else:
            frame_idx+=1
    
    return data_by_frame


class Global_Container(object):
    pass

def load_msgpack_gaze(file):
    
    gaze_data = []
    with open(file, "rb") as fh:
        for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):

            #each row is a packed dict containing topic, norm_pos, conf, ts, recv_ts
            data = msgpack.unpackb(payload, raw=False)                        
            gaze_data.append(data)

    return gaze_data

def plot_norm_pos(data):

    plt.figure(2)    
    data = data[1000:5000] #pick a slice.
    for d in data:        
        npos = d['norm_pos']
        plt.plot(npos[0],npos[1], 'b.', alpha = .4)     
    plt.show()


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
    camera['camera_matrix'] = K
    camera['rect_camera_matrix'] = cm
    camera['dist_coefs'] = D
    camera['resolution'] = resolution
    
    return(camera, rmap)



def D1_undistort_norm_pos(gaze_data):
    
    """uses pickled file to to undistort gaze data"""    
    pickled_file = 'sdcalib.rmap.full.camera_unix.pickle'
    camera, rmap = load_pickled_camera(pickled_file)
    
    K, D, resolution, cm = camera['camera_matrix'], camera['dist_coefs'], camera['resolution'], camera['rect_camera_matrix']

    undistort_gaze_data = []
    for gaze_frame in gaze_data: #each entry may contain multiple gaze estimates.

        if gaze_frame == []: continue    
        if gaze_frame['topic'] == 'gaze.2d.01.':            
            norm_pos = np.array(gaze_frame['norm_pos'])        
            refs = norm_pos*resolution                         
            refs = np.float32(refs).reshape(-1,1,2)                    
            gaze = cv2.fisheye.undistortPoints(refs,K,D, P=K).reshape(2)        
            gaze /= resolution                    
            gaze_frame["norm_pos"] = (float(gaze[0]), float(gaze[1]))                                 
            undistort_gaze_data.append(gaze_frame)
        
    return undistort_gaze_data

def undistort_norm_pos(cam, gaze_data):

    """uses camera model.py to undistort gaze data"""

    def undistort_points(g):

        norm_pos = np.array(g['norm_pos'])
        refs = norm_pos*cam.resolution                         
        refs = np.float32(refs).reshape(-1,1,2)            
        gaze = cv2.fisheye.undistortPoints(refs,K,D, P=K).reshape(2)
        gaze /= cam.resolution                        
        g["norm_pos"] = (float(gaze[0]), float(gaze[1]))     
        return(g)


    K = cam.K
    D = cam.D
    
    for gaze_frame in gaze_data: #each entry may contain multiple gaze estimates.
        if gaze_frame == []: continue
        if type(gaze_frame) is dict:
            gaze_frame = undistort_points(gaze_frame)
        else:
            for g in gaze_frame:            
                g = undistort_points(g)        
                                                            
        
    return gaze_data

def D1_add_viz_trial_timestamps(steeringfolder, pupil_trial_ts):

    """loop through trials and get first and last timestamps"""

    trial_list = pupil_trial_ts['trialcode']
    for index, d in pupil_trial_ts.iterrows():
        fn = steeringfolder + '/' + d['trialcode'] +'.csv'
        vizd = pd.read_csv(fn)
        ts = vizd.currtime.values        
        pupil_trial_ts.loc[index, 'vizt1'] = ts[0]
        pupil_trial_ts.loc[index, 'vizt2'] = ts[-1]                
    
    return(pupil_trial_ts)

def D1_predict_viz_timestamp(trial_ts):
    """
    regress pupil ts onto viz ts to get smooth viz timestamp for gaze.
    """

    pupil_ts = np.concatenate([trial_ts.t1.values, trial_ts.t2.values]).astype(np.float64)
    viz_ts = np.concatenate([trial_ts.vizt1.values, trial_ts.vizt2.values]).astype(np.float64)

    pupil_to_viz = np.polyfit(pupil_ts, viz_ts, 1)
    return(np.poly1d(pupil_to_viz))


def D1_retrieve_pupil_trial_timestamps(pupil_data):

    notes = pupil_data["notifications"]
    trial_timestamps = pd.DataFrame(columns = ['trialcode', 'block', 'pp','sectiontype','mode','condition','count','t1','t2']) 
    entry = 0
    
    for n in notes:        
        if n['subject'] == 'annotation':
            #collect trial type and timestamp            
            
            t = n['timestamp']
            label = n['label']            
            #print(label)            
#            Retrieve trial code.            
            i1 = label.find(' ')            
            i2 = label.find('.')
            if i2 == -1:
                trialcode = label[i1+1:]    
            else:
                trialcode = label[i1+1:i2]                        
            
            reason = label[:i1] #get reason for timestamp.
            
            if trialcode[0] == 'E': continue #errordump trial. Do not record timestamps.
            
           # print(trialcode)
            
            block, pp, sectiontype, condition, count = trialcode.split('_')
                        
           # print(reason)
        
            # check if the trialcode is already in the database. I want one row per trialcode. 
            mask =  trial_timestamps[trial_timestamps['trialcode'].str.contains(trialcode)]
           # print("Mask: ", mask)
           # print("MaskEmpty: ", mask.empty)
            if not mask.empty:
                # if entry already exists, add the timestamp to the relevant column.
                idx = trial_timestamps.index[trial_timestamps['trialcode']==trialcode]
                    
                if "Sta" in reason:
                    trial_timestamps.loc[idx,'t1'] = t
                elif "Fin" in reason:
                    trial_timestamps.loc[idx,'t2'] = t
                                
            else:                 
                # create new entry. 
                mode = "Z"
                if "Man" in reason:
                    mode = "M"
                elif "Aut" in reason:
                    mode = "A"
                elif "PID" in reason:
                    mode = "P"
                elif "Int" in reason:
                    mode = "I"
                
                if "Sta" in reason:
                    t1 = t
                    t2 = 0
                elif "Fin" in reason:
                    t1 = 0
                    t2 = t
                        
                row = [trialcode, block, pp, sectiontype, mode, condition, count, t1, t2]    
                trial_timestamps.loc[entry,:] = row
                entry += 1    

    return(trial_timestamps)


def predict_viz_timestamp(gaze_recalib_data):
    """
    regress onto recalib file.

    """
    #print(gaze_recalib_data)
    recv_ts = []
    pupil_ts = []
    for gaze in gaze_recalib_data:
        
        if gaze == []:continue        
        recv_ts.append(gaze[0]['recv_ts'])
        pupil_ts.append(gaze[0]['timestamp'])

    ten = int(len(pupil_ts) * .1)
    pupil_to_recv = np.polyfit(pupil_ts[ten:-ten], recv_ts[ten:-ten], 1)
    #new_recv = np.poly1d(pupil_to_recv)(pupil_ts)

    return(np.poly1d(pupil_to_recv))
    

def create_gaze_csv(rec_dir, surf, gazefile, steerfile, rerun):
    
    #copies the surface file from the gaze_on_surface directory to the rec directory
    if not os.path.exists(rec_dir + 'surface_definitions'): #if the folder doesn't have a surface definitions file.
        if os.path.exists(surf):
            copyfile(surf, rec_dir + 'surface_definitions')
        else:
            logging.warning("invalid surface file path")        
    
    #1) Load Data
    print('loading gaze')
    if rerun:
        gaze_data = load_msgpack_gaze(os.path.join(rec_dir, gazefile))
    else:
        pupil_data = load_object(os.path.join(rec_dir, gazefile))
        gaze_data = pupil_data['gaze_positions']    


    #2) Undistort gaze
    print("undistorting gaze")
    if rerun:
        screen_size = 1920, 1080
    else:
        screen_size = 1280, 720 #resolution of world camera

    fisheye = camera_models.load_intrinsics(directory=".", cam_name='Pupil Cam1 ID2', resolution = screen_size)
    if rerun:
        gaze_data = undistort_norm_pos(fisheye, gaze_data)
    else:
        gaze_data = D1_undistort_norm_pos(gaze_data)    
    
    #3) Correlate data
    # For dataset one we don't have pupildump of both time estimates, so need to retrieve pupil and viz timestamps for trials from annotations [which contain both].
    #for rerun the cache & timestamps is one directory jumps upwards.
    if rerun:
        sqm_path = os.path.split(rec_dir[:-1])[0] #root folder where square marker cache and timestamps are
    else:
        sqm_path = rec_dir
    timestamps = np.load(os.path.join(sqm_path, "world_timestamps.npy"))            
    gaze_data = correlate_data(gaze_data, timestamps)    
    
    #4) Get timestamps. For dataset1 we don't have pupildump of both time estimates, so need to retrieve pupil and viz timestamps for trials from annotations [which contain both].
    
    if not rerun:
        print("collecting pupil timestamps")
        pupil_trial_ts = D1_retrieve_pupil_trial_timestamps(pupil_data)         
    
        print("collecting vizard timestamps")
        pupil_trial_ts = D1_add_viz_trial_timestamps(steerfile, pupil_trial_ts)                

    #5) predict smooth viz timestamps
    print("predicting timestamps")
    if rerun:
        pupil_to_viz = predict_viz_timestamp(gaze_data)
    else:
        pupil_to_viz = D1_predict_viz_timestamp(pupil_trial_ts)
    

    #6) set up objects and surface tracker for locating gaze on surface.
    g_pool = Global_Container()
    g_pool.rec_dir = rec_dir
    g_pool.timestamps = timestamps
    g_pool.gaze_positions_by_frame = gaze_data

    surface_tracker = Offline_Surface_Tracker(g_pool)        
    persistent_cache = Persistent_Dict(os.path.join(sqm_path,'square_marker_cache'))        
    cache = persistent_cache.get('marker_cache',None)

    
    if rerun:
        camera = load_object('camera')            
        camera = hack_camera(camera)
    else:
        camera, rmap = load_pickled_camera('sdcalib.rmap.full.camera_unix.pickle')
        camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])   

    #7) loop through marker cache and locate gaze on surface    
    print('mapping gaze to surface')
    rows = 0
    mfromscreen = 0
    date = datetime.date(datetime.now())
    filename = 'gaze_on_surface_' + str(date) + '.csv'
    filepath = os.path.join(rec_dir, filename)
    if os.path.exists(filepath): os.remove(filepath) #means multiple runs on a single day
    for s in surface_tracker.surfaces:        
        with open(filepath,'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')            
            csv_writer.writerow(('world_timestamp','world_frame_idx','gaze_timestamp','viz_timestamp','x_norm','y_norm','on_srf','confidence','recalib_std_x','recalib_std_y'))    
            for idx, c in enumerate(cache):
                s.locate(c, camera, 0, 0.0)
                ts = timestamps[idx]             
                if s.m_from_screen is not None:                    
                    mfromscreen += 1
                    """
                    for debugging, the function snake is offline_reference_surface.gaze_on_srf_by_frame_idx -> 
                    reference_surface.map_data_to_surface -> 
                    reference_surface.map_datum_to_surface.

                    """      
                    
                    for gp in s.gaze_on_srf_by_frame_idx(idx,s.m_from_screen):                        
                        rows += 1                        
                        
                        recv_ts = pupil_to_viz(gp['timestamp']) #edited on 02/09/19 for use of 3D calib.
                        
                        if rerun:
                            recalib_std_x, recalib_std_y = gp['recalib_std']
                        else:
                            recalib_std_x, recalib_std_y = [], []

                        #csv_writer.writerow( (ts,idx,gp['timestamp'],gp['recv_ts'],gp['norm_pos'][0],gp['norm_pos'][1], gp['on_srf'], gp['confidence']))#,trialtype))
                        csv_writer.writerow( (ts,idx,gp['timestamp'],recv_ts,*gp['norm_pos'], gp['on_srf'], gp['confidence'],recalib_std_x, recalib_std_y))#,trialtype))
                        
                        #check whether it's writing rows.
                        #print ("called")            
            print("{} rows of data written.".format(rows))
            print("m from screen:", mfromscreen)

            
def main(gazedir, surf, gazefile, steerfile, rerun):
    #rootdir = sys.argv[1]

    rootdir = gazedir        
    #surf = sys.argv[2]
    #surf = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/surface_definitions"

    for dirs in os.walk(rootdir):        
        
        path = str(dirs[0]) + "/"
        print(path)
        
        folder_skip = "2B_reviewed"
        if folder_skip in path: continue
        if os.path.exists(path + gazefile):
            
            #if os.path.exists(path+ "gaze_on_surface.csv"): continue #make sure not already processed
            start = timer()
            print (path)
            create_gaze_csv(path, surf, gazefile, steerfile, rerun)
            compute_time = timer()-start
            print("CSV build took %f seconds" % compute_time)
            #print (path)
            #main(path, surf)

if __name__ == '__main__':

    #gazedir = "C:/git_repos/sample_gaze_Trout"
    gazedir = "F:/Edited_Trout_rerun_Gaze_Videos/Trout_505_2/"
    surf = "C:/git_repos/Trout18_Analysis/Processing/surface_definitions"
    #gazefile = "gaze_recalib_normed.pldata"
    gazefile = "Default_Gaze_Mapper-d82c07cd.pldata"
    
    main(gazedir, surf, gazefile)