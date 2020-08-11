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

def correlate_data(data,timestamps):
    
    """
    timestamps is taken from world_timestamps.


    """   
    timestamps = list(timestamps)
    data_by_frame = [[] for i in timestamps]

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


class Global_Container(object):
    pass

def load_msgpack_gaze(file):
    
    gaze_data = []
    with open(file, "rb") as fh:
        for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):

            #each row is a packed dict containing topic, norm_pos, conf, ts
            data = msgpack.unpackb(payload, raw=False)                        
            gaze_data.append(data)

            return gaze_data

def main(rec_dir, surf, gazefile):
    
    #copies the surface file from the gaze_on_surface directory to the rec directory
    if not os.path.exists(path + 'surface_definitions'): #if the folder doesn't have a surface definitions file.
        if os.path.exists(surf):
            copyfile(surf, path + 'surface_definitions')
        else:
            logging.warning("invalid surface file path")
    
    
    #read gaze calib file.
    #read in jami's gaze.recalib.pldata file 
    gaze_data = load_msgpack_gaze(os.path.join(rec_dir, gazefile))
 
    
  #  print("gaze_recalib_ts", gaze_ts[:10])
    timestamps = np.load(os.path.join(rec_dir, "world_timestamps.npy"))        

 #   print("timestamps", timestamps[:10])
    
    g_pool = Global_Container()
    g_pool.rec_dir = rec_dir
    g_pool.timestamps = timestamps

    #offline reference surface.gaze_on_srf_by_frame_idx funct uses .gaze_positions_by_frame
    g_pool.gaze_positions_by_frame = correlate_data(gaze_data, g_pool.timestamps)

#    print("gpos by frame", g_pool.gaze_positions_by_frame[:10])


    surface_tracker = Offline_Surface_Tracker(g_pool)
    persistent_cache = Persistent_Dict(os.path.join(rec_dir,'square_marker_cache'))
    cache = persistent_cache.get('marker_cache',None)

    mymodel = camera_models.load_intrinsics(directory="..", cam_name='Pupil Cam1 ID2', resolution=(1920, 1080))
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
    camera['camera_matrix'] = np.array(camera['camera_matrix'])
    camera['resolution'] = np.array(camera['resolution'])
    rows = 0
    mfromscreen = 0
    for s in surface_tracker.surfaces:
      # print (s.m_from_screen)
        #surface_name = '_'+s.name.replace('/','')+'_'+s.uid
        #surface_name = '_Corrected'
        
        with open(os.path.join(rec_dir,'gaze_on_surface.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(('world_timestamp','world_frame_idx','gaze_timestamp','x_norm','y_norm','confidence'))    
            for idx, c in enumerate(cache):
                s.locate(c, camera, 0, 0.0)
                ts = timestamps[idx]       
                if s.m_from_screen is not None:                    
                    mfromscreen += 1

                    print("index", idx)

                    gaze_pos =s.gaze_on_srf_by_frame_idx(idx,s.m_from_screen)

                    print("gazeonsrf", gaze_pos)

                    hehe
                    for gp in s.gaze_on_srf_by_frame_idx(idx,s.m_from_screen):                        
                        rows += 1                        
                        print("gp", gp)

                        hehe
                        #need to save ID so I can partial out the eyes for jami's algorithm. 0=left, 1=right
                        csv_writer.writerow( (ts,idx,gp['timestamp'],gp['norm_pos'][0],gp['norm_pos'][1], gp['confidence']))#,trialtype))
                        #check whether it's writing rows.
                        #print ("called")            
            print("{} rows of data written.".format(rows))
            print("m from screen:", mfromscreen)

            hehe
if __name__ == '__main__':
    #rootdir = sys.argv[1]


    """ 
    I'm using gaze.recalib.pldata for the gaze positions, which has been created from the dumped json data using viz.tick(). 

    This means it could be slightly out of sync with the square marker cache, which has the world video file timestamp. 
    
    The world timestamp is reset to viz.tick() at the start of the recording but there will be some drift over time.

    TODO: check with jami the sync of square marker cache and gaze positions. 
    """
    
    rootdir = "E:\\Trout_rerun_gazevideos"         
        
    #surf = sys.argv[2]
    #surf = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/surface_definitions"
    surf= "C:\\git_repos\\Trout18_Analysis\\Processing\\surface_definitions"
    
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"
        if os.path.exists(path + "gaze.recalib.pldata"):
            
            if not os.path.exists(path+ "gaze_on_surface.csv"): #make sure not already processed
                start = timer()
                print (path)
                main(path, surf)
                compute_time = timer()-start
                print("CSV build took %f seconds" % compute_time)
                #print (path)
                #main(path, surf)
