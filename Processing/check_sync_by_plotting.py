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

    """function does not work properly. See extract_markers or overlay script for correct undistortion"""

    data = gaze_data
    for d in data:
        
       
        norm_pos = np.array(d['norm_pos'])
       
        refs = norm_pos*cam.resolution 
       
        refs = cam.unprojectPoints(refs, normalize=False)[:,:-1] #drop the last dimension as only dealing with 2D points.

        refs += 1.0; refs /= 2.0 #rescale to keep in 0,1 for surface tracker.
        
        d['norm_pos'] = np.squeeze(refs)
        
        #print("check attribute", d['norm_pos'])
                
    return data


def select_steergaze_row(recv_ts, steergaze_df):

    viz_ts = steergaze_df['currtime'].values
    #print("recv_ts", recv_ts)
    #print("max viz", max(viz_ts))
    min_idx = viz_ts.searchsorted(recv_ts)
    #print("min idx", min_idx)
    #differences = viz_ts - recv_ts
    #min_idx = np.argmin(differences)
    datarow = steergaze_df.iloc[min_idx]
    return(datarow)

if __name__ == '__main__':

    #rootdir = "E:/Trout_rerun_gazevideos"   
    #rootdir = "C:/git_repos/sample_gaze_Trout/Trout_501_3"
    #rootdir = "C:/git_repos/sample_gaze_Trout/"
    #rootdir = "F:/Edited_Trout_rerun_Gaze_Videos/Trout_510_1"
    rootdir = "F:/Edited_Trout_rerun_Gaze_Videos/Trout_509_2"
    #steergazefile = "C:/git_repos/Trout18_Analysis/Data/steergazedata_reprocessed_050919.csv"
    steergazefile = "C:/git_repos/Trout18_Analysis/Data/GazeAndSteering_newlatency_tidied_3D.csv"
    datafolder = os.path.split(steergazefile)[0]    

    marker_filename = 'square_marker_cache'

    video_res = 1920, 1080

    LATENCY = .12 #estimated lag

    """notes

    GOOD
    
    505_2 - latency of .1 is spot on.
    505_3 - .1 v. good
    504_1 - .125
    506_1 - .14
    507_2 - .07 is spot on.
    507_3 - .06 is spot on.
    508_2 - .125 
    509_1 - .12
    509_2 - .125
    510_3 - .125
    511_1 - .125
    203_2 - .125
    501_3 - .125

    #latencies appear to be somewhat consistent across participant blocks.
    latencies = {
        "501" : .125,
		"502" : .125,
		"503" : .125,
		"504" : .125,
		"505" : .1,
		"506" : .14,
		"507" : .065,
        "508" : .125,
        "509" : .12,
        "510" : .125,
        "511" : .125
	}
    
    AVERAGE
    502_1 - latency of .125 is good.
    503_1 - .125 is good
    
    

    BAD

    """
    
    print("Latency", LATENCY)
        #print(pfolder)
    camera_spec = camera_models.load_intrinsics(directory="", cam_name='Pupil Cam1 ID2', resolution=video_res)

    camera = load_object('camera')        
    camera = hack_camera(camera)

    marker_outline_viz = np.array([[dp.bottom_left_pix,dp.bottom_right_pix,dp.top_right_pix,dp.top_left_pix,dp.bottom_left_pix]],dtype=np.float32)
    #marker_outline_viz = np.array([[bottom_left_pix,bottom_right_pix,top_right_pix,top_left_pix,bottom_left_pix]],dtype=np.float32)

    screenshot_path = "C:/git_repos/Trout18_Analysis/Processing/screenshot_markers.bmp"
    screenshot = cv2.imread(screenshot_path,1)


    #load data with hardcoded trial.
    steergaze = pd.read_csv(steergazefile)
    #steergaze = steergaze.loc[steergaze['ID']==501 and steergaze['Block']==3,:]
#    steergaze = steergaze.query("ID == 501 and Block==3")
    #steergaze = steergaze.query("ID == 510 and block == 1")
    steergaze = steergaze.query("ID == 509 and block == 2")
    steergaze.sort_values('currtime', inplace=True)


    ##load track
    track = pd.read_csv(datafolder + "/track_with_edges.csv")
    inside_edge = track['insidex'].values, track['insidez'].values
    outside_edge = track['outsidex'].values, track['outsidez'].values

    ##load targets
    track = pd.read_csv(datafolder + "/track_with_edges.csv")
    inside_edge = track['insidex'].values, track['insidez'].values
    outside_edge = track['outsidex'].values, track['outsidez'].values

    #load target positions
    targets = pd.read_csv(datafolder + "/TargetPositions.csv")
        
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

            path_3D = path + "/offline_data/gaze-mappings"
            gazedata = pd.read_csv(os.path.join(path_3D, 'gaze_on_surface_3D.csv'))

            start_idx = gazedata['world_frame_idx'].values[0]
            #start_idx +=100 #approximately the first slalom
            #start_idx +=1850 #approximately the first slalom # for 509_1
            #start_idx +=420 #approximately the first slalom # for 509_1
            start_idx +=1900 #approximately the first slalom # for 509_1

            marker_cache = persistent_cache.get('marker_cache',None)

            #camera = load_object('camera')
            #camera = hack_camera(camera)            

            video = cv2.VideoCapture(os.path.join(path,video_file))
            #print ("Opened: ",video.isOpened())        

            """
            gazefile = 'Default_Gaze_Mapper-d82c07cd.pldata'
            #load raw gaze file in screen normalised positions.
            raw_gaze = load_msgpack_gaze(os.path.join(path_3D, gazefile))
            raw_gaze = undistort_norm_pos(camera_spec, raw_gaze)
            world_timestamps = np.load(os.path.join(path, "world_timestamps.npy"))      
            raw_gaze = correlate_data(raw_gaze, world_timestamps)
            """

            #print(raw_gaze)

            print ("starting plotting...")

            """
            TODO:

            use reference_surface.gl_draw_frame.py for the perspective transform code.

            re-estimate homography using marker positions in pixels and the surface definition in order to map screen coords to surface coordinates

            """
            ###make surface
            #TODO: check wherther I can use the deprecated surface definitions.
            surface_definitions = Persistent_Dict(os.path.join(path,'surface_definitions_deprecated'))

            #pprint(surface_definitions)        

            marker_definitions = surface_definitions.get('realtime_square_marker_surfaces')[0]
            #marker_definitions = surface_definitions.get('surfaces')[0]
            g_pool = Global_Container()
            g_pool.rec_dir = path
            s = Offline_Reference_Surface(g_pool, saved_definition=marker_definitions)
            
            #retrieve video.
            marker_frame = None
            next_frame_world_idx = 0
            #prev_frame = None
            for idx, marker_frame_ in enumerate(marker_cache):
                
                if idx < start_idx: continue
                if idx < next_frame_world_idx: continue #skip sections.
                
                print(idx)
                #select closest steering data timestamp.
                
                closest_recv_timestamp = gazedata.loc[gazedata['world_frame_idx']==idx, 'recv_timestamp']                                
                closest_recv_timestamp = np.mean(closest_recv_timestamp)
                if np.isnan(closest_recv_timestamp): continue
                
                #manual correction. minus to delay steering data.
                closest_recv_timestamp -= LATENCY

                #print('recv_ts', closest_recv_timestamp)
                row = select_steergaze_row(closest_recv_timestamp, steergaze)
                #print(row)
                #print('viz_ts', row['currtime'])

                #check that the next timestamp follows on, or whether there is missing data.
                #find index in gaze data. If massively off, set skip_idx
                recv_ts_array = gazedata['recv_timestamp'].values
                next_steering_frame_idx = recv_ts_array.searchsorted(row['currtime'])
                next_frame_world_idx = gazedata.loc[next_steering_frame_idx, 'world_frame_idx']          

                #print('gaze frame idx', idx)
                #print('next steering frame idx', next_frame_world_idx)

                #print('diff', idx - next_frame_world_idx)                
                
                
                #pick target positions
                condition = row['condition']
                target_centres = targets.loc[targets['condition']==condition]
                target_circles = dp.target_position_circles(target_centres) #create position arrays
                
                yaw = row['yaw']
                viewpoint = row['posx'], row['posz']
               # gaze_on_screen = row['hangle_new'], row['vangle_new']

                #plot track
                inside_edge_pixels = dp.world_to_screen_homo_cave(np.transpose(inside_edge), viewpoint, yaw)
                outside_edge_pixels = dp.world_to_screen_homo_cave(np.transpose(outside_edge), viewpoint, yaw)

                #remove any above the horizon. These are out of view.               
                #inside_edge_pixels = inside_edge_pixels[inside_edge_pixels[:,1]<cy, :]
                #outside_edge_pixels = outside_edge_pixels[outside_edge_pixels[:,1]<cy, :]                                
                
                plt.plot(inside_edge_pixels[:,0], video_res[1] - inside_edge_pixels[:, 1], 'k.', markersize = .5)
                plt.plot(outside_edge_pixels[:,0], video_res[1] - outside_edge_pixels[:, 1], 'k.', markersize = .5)

                ##plot targets 
                #compute target arrays from viewpoint
                for target in target_circles:

                    target_circle = np.squeeze(np.array(target))
                    print("target_circle", target_circle.shape)
                    target_pixels = dp.world_to_screen_homo_cave(np.transpose(target_circle), viewpoint, yaw)

                   # target_pixels = target_pixels[target_pixels[:,1]<cy,:]

                    if condition in [0,1]:
                        mycolour = 'b.'
                    elif condition in [2,3]:
                        mycolour = 'r.'

                    
                    plt.plot(target_pixels[:,0], video_res[1] - target_pixels[:,1], mycolour, markersize = .2)

                #plot gaze
                #take gaze from gaze_csv.
                srf_x = np.mean(gazedata.loc[gazedata['world_frame_idx']==idx, 'x_norm'])
                srf_y = np.mean(gazedata.loc[gazedata['world_frame_idx']==idx, 'y_norm'])
                #srf_x, srf_y = row['x_norm'], row['y_norm']
                screen_coords = dp.surface_to_screen([srf_x, srf_y])
                screen_coords *= video_res


                if row['confidence'] > .4: #generous
                    
                    plt.plot(screen_coords[0], video_res[1] - screen_coords[1], 'mo', markersize = 5)
                else:
                    pass    

                video.set(cv2.CAP_PROP_POS_FRAMES, idx) #set to new frame for the sake of missing data.

                #plot video.
                ret, oframe = video.read()

                frame = camera_spec.undistort(oframe)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                marker_frame = square_marker_detect.detect_markers_robust(frame, 5, marker_frame)
                s.locate(marker_frame, camera, 0, 0.0)

                surface_outline = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32)            

                #https://stackoverflow.com/questions/45817325/opencv-python-cv2-perspectivetransform
                #perspectiveTransform needs two channels.
                try:
                    surface_outline_vid = cv2.perspectiveTransform(surface_outline,s.m_to_screen)
                except: continue

                surface_outline_vid *= video_res  #in pixels

                #marker outline need to be sorted counter-clockwise stating at bottom left
                h, status = cv2.findHomography(surface_outline_vid, marker_outline_viz)
                surface_outline_viz = cv2.perspectiveTransform(surface_outline_vid, h)

                frame = cv2.flip(frame, 0)
                warp = cv2.warpPerspective(frame, h, frame.shape[::-1])
            
                surface_outline_viz = np.squeeze(surface_outline_viz)            

                plt.plot(surface_outline_viz[:,0],video_res[1] - surface_outline_viz[:,1], 'g-') #flip.
                
                """
                #plot markers
                for marker in marker_frame:
                    
                    verts = marker.get('verts')

                    for vertices in verts:

                        vertex = vertices[0]

                        plt.plot(vertex[0], video_res[1] - vertex[1], 'ro', markersize = 2)
                        plt.ylim(0, video_res[1])
                        plt.xlim(0, video_res[0])

                """
                #plt.imshow(cv2.flip(frame, 0), cmap='gray')
                #plt.ylim(0, video_res[1])
                #plt.xlim(0, video_res[0])
                plt.imshow(cv2.flip(warp,0))
                #plt.imshow(screenshot)
                plt.pause(.016)                    
                plt.cla()
            plt.show()   
                 

                    