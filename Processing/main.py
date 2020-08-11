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

def correlate_data(data,timestamps):
    
        
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


def main(rec_dir, surf):
    
    #copies the surface file from the gaze_on_surface directory to the rec directory
    if not os.path.exists(path + 'surface_definitions'): #if the folder doesn't have a surface definitions file.
        if os.path.exists(surf):
            copyfile(surf, path + 'surface_definitions')
        else:
            logging.warning("invalid surface file path")
    
     pupil_data = load_pldata_file(path, 'gaze.recalib.pldata')
    
    ##Split up file according to annotations. 
    
    notes = pupil_data["notifications"]
    trial_timestamps = pd.DataFrame(columns = ['trialcode', 'block', 'pp','sectiontype','mode','condition','count','t1','t2']) 
    entry = 0
    for n in notes:
        if n['subject'] == 'annotation':
            #collect trial type and timestamp
            
            print(n)
            
            t = n['timestamp']
            label = n['label']
            
#            Retrieve trial code.
            
            i1 = label.find(' ')            
            i2 = label.find('.')
            if i2 == -1:
                trialcode = label[i1+1:]    
            else:
                trialcode = label[i1+1:i2]                        
            
            reason = label[:i1] #get reason for timestamp.
            
            if not trialcode[0] == 'E': #errordump trial. Do not record timestamps.
            
                print(trialcode)
                
                block, pp, sectiontype, condition, count = trialcode.split('_')
                            
                print(reason)
    
            
                # check if the trialcode is already in the database. I want one row per trialcode. 
                mask =  trial_timestamps[trial_timestamps['trialcode'].str.contains(trialcode)]
                print("Mask: ", mask)
                print("MaskEmpty: ", mask.empty)
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
                                    
    #mynote = notes[-1] #annotation is always at the end. 
    #trialtype = mynote['label']
    #print (notes)
    #print('trialtype: ', trialtype) 
    print (trial_timestamps)
    
    trial_timestamps.to_csv(rec_dir + "Eyetrike_trialtimestamps.csv")
    pupil_list = pupil_data['pupil_positions']
    gaze_list = pupil_data['gaze_positions']
    timestamps = np.load(os.path.join(rec_dir, "world_timestamps.npy"))
    
    g_pool = Global_Container()
    g_pool.rec_dir = rec_dir
    g_pool.timestamps = timestamps
    
    #error seems to be here, as this returns empty. 
    g_pool.gaze_positions_by_frame = correlate_data(gaze_list, g_pool.timestamps)
    #print (g_pool.gaze_positions_by_frame)
           # print (g_pool.gaze_positions_by_frame)

    surface_tracker = Offline_Surface_Tracker(g_pool)
    persistent_cache = Persistent_Dict(os.path.join(rec_dir,'square_marker_cache'))
    cache = persistent_cache.get('marker_cache',None)
    camera = load_object('camera')
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
    camera['camera_matrix'] = np.array(camera['camera_matrix'])
    camera['resolution'] = np.array(camera['resolution'])
    rows = 0
    mfromscreen = 0
    for s in surface_tracker.surfaces:
      # print (s.m_from_screen)
        surface_name = '_'+s.name.replace('/','')+'_'+s.uid
        surface_name = '_Corrected'
        
        with open(os.path.join(rec_dir,'gaze_on_surface'+surface_name+'.csv'),'w',encoding='utf-8',newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(('world_timestamp','world_frame_idx','gaze_timestamp','x_norm','y_norm','on_srf','id','confidence'))    
            for idx, c in enumerate(cache):
                s.locate(c, camera, 0, 0.0)
                ts = timestamps[idx]       
                if s.m_from_screen is not None:                    
                    mfromscreen += 1
                    for gp in s.gaze_on_srf_by_frame_idx(idx,s.m_from_screen):                        
                        rows += 1                        
                        #need to record which eye makes up the gaze position for duplicate timestamps. This means that later on we can average across opposite eyes.
                        if len((gp['base_data']['base_data'])) > 1:
                            eye_id = 2 #means both eyes were involved.
                        else:
                            eye_id = gp['base_data']['base_data'][0]['id'] #id of the single eye gaze position. Can be 0=left or 1=right.
                        #need to save ID so I can partial out the eyes for jami's algorithm. 0=left, 1=right
                        csv_writer.writerow( (ts,idx,gp['base_data']['timestamp'],gp['norm_pos'][0],gp['norm_pos'][1],gp['on_srf'],eye_id, gp['confidence']))#,trialtype))
                        #check whether it's writing rows.
                        #print ("called")
            print("{} rows of data written.".format(rows))
            print("m from screen:", mfromscreen)
if __name__ == '__main__':
    #rootdir = sys.argv[1]
    
    rootdir = "E:\\Trout_rerun"         
        
    #surf = sys.argv[2]
    #surf = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/surface_definitions"
    surf= "C:\\git_repos\\Trout18_Analysis\\Processing\\surface_definitions"
    
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"
        if os.path.exists(path + "gaze.recalib.pldata"):
            
            if not os.path.exists(path+ "gaze_on_surface_Corrected.csv"): #make sure not already processed
                start = timer()
                print (path)
                main(path, surf)
                compute_time = timer()-start
                print("CSV build took %f seconds" % compute_time)
                #print (path)
                #main(path, surf)
