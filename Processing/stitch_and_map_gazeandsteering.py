###script to view and almagamate eye tracking files.
import numpy as np
import sys, os
from file_methods import *
import pickle
import matplotlib.pyplot as plt
import cv2
import csv
import pandas as pd
import math as mt 
from scipy.interpolate import interp1d
from nslr_hmm import *
from timeit import default_timer as timer
from glob import glob
from pathlib import Path
from pprint import pprint
import calhelper as ch

import drivinglab_projection as dp
from TrackMaker import TrackMaker

screen_res = 1920, 1080

"""
Useful docs:
    
To understand marker transform:    
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
   
    #Screen co-ords.
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+

    
"""
    
def add_all_coord_systems(df):

    """adds columns for surface, display screen_normed, display screen_pixels, angles."""

    #start with surface_norms
    srf_x = df['x_norm'].values
    srf_y = df['y_norm'].values

    #add screen_coordinates for both norm and pixels.
    screen_coords_norm = dp.surface_to_screen(np.transpose(np.array([srf_x, srf_y])))
    df['screen_x_norm'] = screen_coords_norm[:,0]
    df['screen_y_norm'] = screen_coords_norm[:,1]
    
    screen_coords_pix = screen_coords_norm*screen_res
    df['screen_x_pix'] = screen_coords_pix[:,0]
    df['screen_y_pix'] = screen_coords_pix[:,1]

    #add angles
    gaze_angles = dp.screen_to_angles(screen_coords_norm)
    df['hangle'] = gaze_angles[:,0]
    df['vangle'] = gaze_angles[:,1]

    return(df)       
    
def LoadSteeringBlock(steerdir, block, pcode):
    #return steering data for trial. 
   # df_steer = pd.DataFrame() #trial frame by frame data
    steering_block = pd.DataFrame() #master data for gaze and steering          

    file_prefix = '_'.join([block, pcode])
    #EXPBLOCK_PP_drivingmode_condition_count
    for fn in glob(steerdir + "/" + file_prefix + "_*.csv"):
        
        #print(fn)
        
        trial_data = pd.read_csv(fn)         

        #this will be the full path. get the file name without the extension for the trialcode field
        split_path = os.path.split(fn)        
        trialcode = os.path.splitext(split_path[-1])[0]        
        trial_data['trialcode'] = trialcode

        #add some more useful identifiers that may not be in there already
        #print("trialcode:", trialcode)
        
        block, ppid, sectiontype, condition, count =  trialcode.split("_") 
        trial_data['count'] = count 
        trial_data['condition'] = condition      
        trial_data['trial_match'] = "_".join([ppid, condition, block, count])
        
        steering_block = pd.concat([steering_block,trial_data])                        
    
        
    return(steering_block)
    
def StitchGazeAndSteering(df_gaze, df_steer, latency):
    
    #linearly interpolate so that timestamps match exactly.
    #recv_timestamp should be viz.tick()

    #interpolate the normed positions
    yinterpolater_ynorm = interp1d(df_gaze['viz_timestamp'].values - latency,df_gaze['y_norm'].values)
    y_norm = yinterpolater_ynorm(df_steer['currtime'].values)
                    
    xinterpolater_xnorm = interp1d(df_gaze['viz_timestamp'].values - latency,df_gaze['x_norm'].values)
    x_norm = xinterpolater_xnorm(df_steer['currtime'].values)
    
    df_steer.loc[:,'y_norm'] = y_norm
    df_steer.loc[:,'x_norm'] = x_norm  


    #interpolate confidence values.
    confidenceinterpolater = interp1d(df_gaze['viz_timestamp'].values - latency,df_gaze['confidence'].values)
    confidence = confidenceinterpolater(df_steer['currtime'].values)

    df_steer.loc[:,'confidence'] = confidence

    return(df_steer)    
    
def GazeinWorld(df, midline, trackorigin):
    
    #convert matlab script to python to calculate gaze from centre of the road, using Gaze Angles.
    
    def GazeMetrics(row):
        
        EH = 1.2
    
       # print("ROW: ", row)
    
        H = row['hangle'] #gaze angle on surface
        V = row['vangle']
        heading_degrees = row['yaw'] #heading in virtual world
        heading_rads = (heading_degrees*mt.pi)/180 #convert into rads.
        xpos = row['posx'] #steering position
        zpos = row['posz']
        #OSB = row['OSB'] #for calculating gaze bias
        
#        if Bend == "Left":
#            H = H * -1 #gaze localisation assumes a right bend
        
        
        ##project angles relative to horizon and vert centre-line through to ground
    
        #mt functions take rads.
#        print ("Vangle: ", V)
        
        hrad = (H*mt.pi)/180
        vrad = (V*mt.pi)/180
    
        zground = -EH/np.tan(vrad) 
        xground = np.tan(hrad) * zground
        lookahead = np.sqrt((xground**2)+(zground**2)) #lookahead distance from midline, before any rotation. Will be in metres.
    
        #rotate point of gaze using heading angle.
        xrotated = (xground * np.cos(-heading_rads)) - (zground * np.sin(-heading_rads))
        zrotated = (xground * np.sin(-heading_rads)) + (zground * np.cos(-heading_rads))
    
        #add coordinates to current world position.
        xpog = xpos+xrotated
        zpog = zpos+zrotated 
    
        #create pythagorean distance array from midline
        gazedist = np.sqrt(((xpog-midline[:,0])**2)+((zpog-midline[:,1])**2)) #array of distances from midline to gaze position in world.        
        idx = np.argmin(abs(gazedist)) #find smallest difference in pythag distance from 0,0 to get closest point.
        closestpt = midline[idx,:]
        dist = gazedist[idx] #distance from midline
        
#			#Sign bias from assessing if the closest point on midline is closer to the track origin than the driver position. Since the track is an oval, closer = understeering, farther = oversteering.                                
        middist_from_origin = np.sqrt(((closestpt[0]-trackorigin[0])**2)+((closestpt[1]-trackorigin[1])**2))  #distance of midline to origin
        pos_from_trackorigin = np.sqrt(((xpog-trackorigin[0])**2)+((zpog-trackorigin[1])**2)) #distance of driver pos to origin
        distdiff = middist_from_origin - pos_from_trackorigin #if driver distance is greater than closest point distance, steering position should be understeering        
        gazebias = dist * np.sign(distdiff)         
#        print (gazebias)
##        
        # print("GazeBias: ", gazebias)
        # print("Lookahead: ", lookahead)
        # print("Xpog", xpog)
        # print("Zpog", zpog)
        
#        plt.plot(range(len(gazedist)),gazedist)
#        plt.show()
#        
#        plt.plot(xmid, zmid, 'b-')
#        plt.plot(xpog,zpog,'r.')
#        plt.plot(xrd,zrd,'g.')
#        plt.plot(xpos,zpos,'k.')
#        plt.axis('equal')
#        plt.show()
        
        return (lookahead, gazebias, xpog, zpog)
    
    df['lookahead'], df['gazebias'], df['xpog'],df['zpog'] = zip(*df.apply(GazeMetrics,axis=1))    
    #row = df.loc[2,:]
    #lookahead, gazebias, xpog, zpog = GazeMetrics(row)
   
    
    return (df)

def GazeonPath_inworld(df):
    
    """Add Distance of Gaze landing point to future vehicle trajectory. Must get called after GazeinWorld since relies on xpog and zpog"""
    
    """
    - Path Distance: The distance of gaze landing point, in metres, relative to the closest future vehicle trajectory position as the actual trajectory passes the gaze landing point.
    
    """

    #load vehicle trajectory before looping through rows.
    Vehicle_posx = df['posx'].values
    Vehicle_posz = df['posz'].values

    def CalculateGazeDistance(row):
        
        #load gaze landing point.
        Gaze_posx = row['xpog']
        Gaze_posz = row['zpog']
        
        gazedistance_array = np.sqrt(((Gaze_posx-Vehicle_posx)**2)+((Gaze_posz-Vehicle_posz)**2)) #array of distances from trajectory to gaze landing point in world.        
        idx = np.argmin(abs(gazedistance_array)) #find smallest difference in pythag distance from 0,0 to get closest point.        
        gazedistance = gazedistance_array[idx] #distance from midline
        
        ##re-add an on_srf boolean.
        
        y_norm = row['y_norm']
        x_norm = row['x_norm']        
        
        on_srf = "TRUE"
        if (y_norm < 1) & (y_norm > 0) & (x_norm < 1) & (y_norm > 0):
            on_srf = "TRUE"
        else:
            on_srf = "FALSE"            

        return (gazedistance, on_srf)
    
    df['gazetopath_metres'], df['on_srf'] = zip(*df.apply(CalculateGazeDistance,axis=1))    
    #row = df.loc[2,:]
    #lookahead, gazebias, xpog, zpog = GazeMetrics(row)
   
    
    return (df)


def GazeonPath_angles(df):
    """calculates gaze to path in angles"""
    traj_x = df['posx'].values #use the unmirrored values
    traj_z = df['posz'].values #use the unmirrored values

    trajectory = np.transpose(np.array([traj_x, traj_z]))

    for index, (_, row) in enumerate(df.iterrows()):
        yaw = row['yaw']
        viewpoint = row['posx'], row['posz']
        gaze_on_screen = row['hangle'], row['vangle']
        #print("gaze", gaze_on_screen)

        #calc gaze to path
        future_traj = trajectory[index:,:]
        traj_angles, depth = dp.world_to_angles_through_screen(future_traj, viewpoint, yaw)        
        traj_angles = traj_angles[depth > 0,:]
        
        if len(traj_angles) > 0:

        #find index of minimum distance.
        
            distances =  np.subtract(np.array(gaze_on_screen),traj_angles) #left of future traj = negative.
            distances = distances[~np.isnan(distances)].reshape(-1,2)                                    
            gazedistance_array = np.sqrt((distances[:,0]**2)+(distances[:,1]**2)) #array of distances from trajectory to gaze landing point in world.                    
            idx = np.argmin(abs(gazedistance_array)) #find smallest difference in pythag distance from 0,0 to get closest point.        
            angular_distance = gazedistance_array[idx]
            h_dist = distances[idx,0]
            v_dist = distances[idx,1]        

            #use the idx of closest angular distance as the value to calculate time headway from        
            #TODO: Compute the path distance and use that instead of pythagoras.
            world_ref_th = future_traj[idx,:]
            ego_distance = np.sqrt((world_ref_th[0] - viewpoint[0])**2 +(world_ref_th[1] - viewpoint[1])**2) 
            ego_timeheadway = ego_distance / 8.0 #8ms is the velocity.
        else:
            angular_distance = np.nan
            h_dist = np.nan
            v_dist = np.nan
            ego_timeheadway = np.nan
            
        df.loc[index,'gazetopath_angulardistance'] = angular_distance
        df.loc[index,'gazetopath_hangle'] = h_dist
        df.loc[index,'gazetopath_vangle'] = v_dist
        df.loc[index,'timeheadway_angularref'] = ego_timeheadway

        """
        #calc closest index of gaze distance so I can plot.
         #load gaze landing point.        
        Gaze_posx = row['xpog']
        Gaze_posz = row['zpog']
        
        real_world_array = np.sqrt(((Gaze_posx-traj_x)**2)+((Gaze_posz-traj_z)**2)) #array of distances from trajectory to gaze landing point in world.        
        w_i = np.argmin(abs(real_world_array)) #find smallest difference in pythag distance from 0,0 to get closest point.        
        world_distance = real_world_array[w_i] #distance from midline

        #corrected index for start of future traj
        w_i -= index
        
        print("real_world_distance", world_distance)
        world_ref = traj_angles[w_i,:]

        ##re-add an on_srf boolean.

        #plot
        
        ref = traj_angles[idx, :]
        
        print("angular_distance", angular_distance)
        print("h_dist", h_dist)
        print("v_dist", v_dist)

        
        angles_limits_bottom = dp.screen_to_angles([0,0])[0]
        angles_limits_top = dp.screen_to_angles([1,1])[0]

        

        plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
        
        plt.xlim(angles_limits_bottom[0],angles_limits_top[0])
        
        plt.plot(traj_angles[:,0],traj_angles[:,1], 'ko', markersize = .5)
        plt.plot(ref[0],ref[1], 'ro', markersize = 5)
        plt.plot(world_ref[0],world_ref[1], 'go', markersize = 5)
        
        plt.plot(gaze_on_screen[0],gaze_on_screen[1], 'mo', markersize = 5)
        
        
        plt.pause(.5)                    
        plt.cla()
                        
    plt.show()
    """
    
    
    return(df)

def GazeToObjects(df, centres, circles):

    """calculates angles and distances for every target encountered in that trial
    
    For each target we want:
    On_screen flag.
    angular centroid
    highest, lowest, left, right points.
    signed angle in two dimensions.
    actual distance.

    """    

    #calculate trajectory for plotting
    traj_x = df['posx'].values #use the unmirrored values
    traj_z = df['posz'].values #use the unmirrored values

    trajectory = np.transpose(np.array([traj_x, traj_z]))


    for index, (_, row) in enumerate(df.iterrows()):
        yaw = row['yaw']
        viewpoint = row['posx'], row['posz']
        hang, vang = row['hangle'], row['vangle']                

        #pprint(circles)

        #calculate trajectory
        future_traj = trajectory[index:,:]
        traj_angles, depth = dp.world_to_angles_through_screen(future_traj, viewpoint, yaw)  
        traj_angles = traj_angles[depth > 0,:]      

        any_on_screen = False
        for i, circle in enumerate(circles):
            colname = 'T' + str(i) + '_'

            circle = np.squeeze(np.array(circle))
            target_pixels, depth = dp.world_to_screen_homo_cave(np.transpose(circle), viewpoint, yaw)
            target_pixels = target_pixels[depth > 0, :]

            on_screen = False
            bool_mask = (target_pixels[:,0] > 0) & (target_pixels[:,0] < screen_res[0]) & (target_pixels[:,1] > 0) & (target_pixels[:,1] < screen_res[1])
            on_screen_pix = target_pixels[bool_mask, :]
            if len(on_screen_pix) > 50: #some arbitrary number that means the target may be visible.
                on_screen = True
                any_on_screen = True

            df.loc[index,colname + 'on_screen'] = int(on_screen)

            #print("on_screen:", on_screen)

            #print("gaze_on_screen", hang, vang)

            """
            #NOT NEEDED FOR TROUT1 PROCESSING

            target_angles, depth = dp.world_to_angles_through_screen(np.transpose(circle), viewpoint, yaw)
            target_angles = target_angles[depth>0, :]
            if len(target_angles) > 0:
                h_min, h_max = min(target_angles[:,0]), max(target_angles[:,0])
                v_min, v_max = min(target_angles[:,1]), max(target_angles[:,1])
            else:
                h_min, h_max, v_min, v_max = np.nan, np.nan, np.nan, np.nan

            df.loc[index,colname + 'h_min_angle'] = h_min
            df.loc[index,colname + 'h_max_angle'] = h_max
            df.loc[index,colname + 'v_min_angle'] = v_min
            df.loc[index,colname + 'v_max_angle'] = v_max

            #print("h_range", h_min, h_max)
            #print("v_range", v_min, v_max)

            #retrieve centres
            #print("centres", len(centres))
            #print("index", i)
            
            centre = centres.iloc[i, :]

            xcentre = centre['xcentre']
            zcentre = centre['zcentre']           

            centroid_angles, depth = dp.world_to_angles_through_screen(np.array([xcentre, zcentre]), viewpoint, yaw)
            centroid_angles = centroid_angles[depth>0,:]

            if len(centroid_angles) > 0:

                centroid_angles = np.squeeze(centroid_angles)
                df.loc[index,colname + 'centroid_hangle'] = centroid_angles[0]
                df.loc[index,colname + 'centroid_vangle'] = centroid_angles[1]


            #print("centroid", centroid_angles)
                h_distance =  hang - centroid_angles[0] #left of future traj = negative.
                v_distance =  vang - centroid_angles[1] #below future traj = positive.
                gazedistance = np.sqrt((h_distance**2)+(v_distance**2)) #array of distances from trajectory to gaze landing point in world.        

                df.loc[index,colname + 'gazetoobject_angulardistance'] = gazedistance
                df.loc[index,colname + 'gazetoobject_hangle'] = h_distance
                df.loc[index,colname + 'gazetoobject_vangle'] = v_distance

                #print("angular distance", gazedistance)
                #print("angular h", h_distance)
                #print("angular v", v_distance)

                world_distance = np.sqrt((xcentre - row['xpog'])**2 + (zcentre - row['zpog'])**2)
                df.loc[index,colname + 'gazetoobject_metres'] =world_distance

            else:
                df.loc[index,colname + 'centroid_hangle'] = np.nan
                df.loc[index,colname + 'centroid_vangle'] = np.nan
                df.loc[index,colname + 'gazetoobject_angulardistance'] = np.nan
                df.loc[index,colname + 'gazetoobject_hangle'] = np.nan
                df.loc[index,colname + 'gazetoobject_vangle'] = np.nan
                df.loc[index,colname + 'gazetoobject_metres'] = np.nan

            #print("world distance", world_distance)
            """

            ####plot centres####
            """
            if on_screen:
                      
                condition = centre['condition']        
                if condition in [0,1]:
                    mycolour = 'b.'
                elif condition in [2,3]:
                    mycolour = 'r.'

                #limits                              
                angles_limits_bottom = dp.screen_to_angles([0,0])[0]
                angles_limits_top = dp.screen_to_angles([1,1])[0]

                plt.ylim(angles_limits_bottom[1],angles_limits_top[1])                
                plt.xlim(angles_limits_bottom[0],angles_limits_top[0])

                #target angles                
                plt.plot(target_angles[:,0],target_angles[:,1], mycolour, markersize = .1)

                #plot centroid
                plt.plot(centroid_angles[0],centroid_angles[1], 'ko', markersize = 4)

                
                
        if any_on_screen:
            #plot future traj
            plt.plot(traj_angles[:,0],traj_angles[:,1], 'go', markersize = .1)

            #plot gaze
            plt.plot(hang,vang, 'mo', markersize = 5)                
            plt.pause(.1)                    
            plt.cla()
                        
    plt.show()
    """

    return(df)

            
def mirror_and_add_roadsection(df):

    """
    moved from tidy_data_analysis.R to remove the need for an extra script.
    adds useful columns and clearer names
    """

    df['currtimezero'] = df.currtime.values - df.currtime.values[0]
    df.rename(columns = {'sectiontype':'drivingmode','Block':'block'}, inplace=True)
    
    #mirror trials.
    nrows = df.shape[0]
    df['f'] = range(1,nrows+1)
    mirror_flag = np.sign(df.posz.values[0])
    df['startingposition'] = mirror_flag
    df['posx_mirror'] = df.posx.values * mirror_flag
    df['posz_mirror'] = df.posz.values * mirror_flag
    df['xpog_mirror'] = df.xpog.values * mirror_flag
    df['zpog_mirror'] = df.zpog.values * mirror_flag

    new_yaw = df.yaw.values
    #mirroring yaw requires resetting those that are above 360 degrees.
    if mirror_flag < 0:
        for yaw in new_yaw:
            yaw += 180
            if yaw > 360: yaw -= 360

    df['yaw_mirror'] = new_yaw


    ## Crop the road sections at 1 s away from the start of the change in road geometry. (i.e. bend starts at z = 60 m, so classify the gaze data as z = 52 m)
    roadsection = np.empty(nrows)
    x_mirrs = df.posx_mirror.values
    z_mirrs = df.posz_mirror.values
    for i, road in enumerate(roadsection):
    
        x_mirr = x_mirrs[i]
        z_mirr = z_mirrs[i]

        if (x_mirr <0) & (z_mirr < 52):
            road = 0
        elif (x_mirr <= 23.7) & (z_mirr >= 52):
            road = 1
        elif (x_mirr > 23.7) & (z_mirr >= 28):
            road = 2
        else:
            road = 99
            
        roadsection[i] = road

    df['roadsection'] = roadsection
    #print("here")

    return(df)

def main(gazedir, steerdir, savedir, outfile, latency = None, rerun = True, date = None):
	#rootdir = sys.argv[1] 
    #rootdir = "E:\\Trout_rerun_gazevideos"    
    #rootdir = "E:\\Trout_rerun_gazevideos"   
    rootdir = gazedir
    
    #df = pd.read_csv("../Data/trout18_gazeandsteering_2019-11-21.csv")
    #existing_trials = set(df.trialcode.values)  
    
    #savedir = "E:/Trout_rerun_processed_gazesteeringdata"
    
    stitchfilepath = savedir + outfile
    stitchfilepath = ch.check_exist(stitchfilepath)
    resave = False #boolean whether to move files to savedir
    #steerdir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Master_SampleParticipants/SteeringData/"
    
   # marker_corners_norm = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
   # marker_box = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32) #this is for perspectiveTransform.
   
    INCLUDE_GAZE = False

    #latency = .175 #estimated lag #visual inspection for check_sync_by_plotting.py
    latencies = {
        "501" : .125,
		"502" : .125,
		"503" : .125,
        "203" : .125,
		"504" : .125,
		"505" : .1,
		"506" : .14,
		"507" : .065,
        "508" : .125,
        "509" : .12,
        "510" : .125,
        "511" : .125
	}
    #CREATE MIDLINE. Use Function from Main Experiment
    TrackData = TrackMaker(10000)
    midline = TrackData[0]     
    trackorigin =TrackData[1]

    ##0 = Attract_Narrow, 1=Attract_Medium, 2=Attract_Wide, 3=Avoid_Narrow, 4=Avoid_Medium, 5=Avoid_Wide
    #cndtcodes = ['Attract_Narrow','Attract_Medium','Attract_Wide','Avoid_Narrow','Avoid_Medium','Avoid_Wide']   
    #cndtcodes = ['Attract_Narrow','Attract_Wide','Avoid_Narrow','Avoid_Wide']      
        
#    master_gaze = pd.DataFrame() #master data for gaze file.
    master_segment = pd.DataFrame() #master data for segment file.                
    master_stitch = pd.DataFrame() #master data for gaze and steering            
    
    
    targetpath = "C:/git_repos/Trout18_Analysis/Data"

    targets = pd.read_csv(targetpath + "/TargetPositions.csv")


    processing_date = date
    if processing_date is None:
        raise Exception("invalid processing date given")
    gazedata_filename = '/gaze_on_surface_' + processing_date + '.csv'

    #For pilot scripts 
    #STEERING_TIME_CORRECTION = 74.60509197 #Take the first timestamp of the first steering trial, since vizard timestamp isn't zeroed after calibration but pupil timestamp is.
    #EYETRACKING_TIME_CORRECTION = 6568.631288

    #gazedata_filename = '/gaze_on_surface.csv'
    #gazedata_filename = '/gaze_on_surface_3D.csv'
    #print(pfolder)
    #print ("here")
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) 
        print (path)
        folder_skip = "2B_reviewed"
        gaze_skip = "gaze-mappings"

        if folder_skip in path: continue
        if gaze_skip in path: continue #do not want to process gaze mappings
        
        if os.path.exists(path +gazedata_filename):                    
            
            """                    
            We have a gaze video for each block. There are four blocks per participant.
            the gaze_on_surface data has the following columns:

            world_timestamp, world_frame_idx,gaze_timestamp, recv_timestamp,	x_norm,	y_norm,	on_srf,	confidence

            recv_timestamp is viz.tick()
            gaze_timestamp is the eye camera timestamp
            world_timestamp is the world video timestamp
                                   
            """                                                               
            begin = timer()

            #load the gaze video file.
            gaze_block_df = pd.read_csv(path+gazedata_filename, sep=',',header=0)                                        

            #retrieve the EXP_ID used.
            path = Path(path)                        
            #ppfolder = path.parts[-2]
            ppfolder = path.parts[-4]

            i = 1
            while True:
                ppfolder = path.parts[-i]
                i +=1
                if 'Trout' in ppfolder:
                    break
#           
            print("ppfolder", ppfolder)            
            expname, pcode, block = ppfolder.split('_')
          #  print(expname)
            print(pcode)
            print(block)

#            if pcode not in ['10', '9']: continue
#            if block == '2': continue

            if rerun:
                latency = latencies[pcode]
            else:
                latency = 0
           # print("latency", latency)
            
            #load the steering block
            steering_block_df = LoadSteeringBlock(steerdir, block, pcode)                                                            

            #pprint(steering_block_df) 
            #now we have the steering_block_df we want to loop through the trials and add the gaze data the corresponds with the timezones for each trial. We do not need to save the calibration data here.                        
            for trialcode, trialdata in steering_block_df.groupby('trialcode'):
                #if trialcode in existing_trials: continue #don't process trials done yesterday.

                #trialcodes_inspect = ['1_10_0_0_0','1_10_0_0_3','2_10_0_1_1', '2_9_0_1_1', '2_9_0_0_1']            
                #if not (trialcode in trialcodes_inspect): continue            
                try:
            
                    print("Processing: ", trialcode)                    
                
                    trialdata = trialdata.copy()

                    #get the timestamps for that particular trial.
                    mint = trialdata['currtime'].values[0] #could also use min and max
                    maxt = trialdata['currtime'].values[-1]

                    #pick the corresponding gaze timestamps
                    lower = gaze_block_df['viz_timestamp'] > (mint - .5) #add half a second buffer so the steering timestamps are within the interpolation rangte.
                    upper = gaze_block_df['viz_timestamp'] < (maxt + .5)
                    gaze_trial_df = gaze_block_df.loc[lower & upper, :].copy() 

                    #HACK for when there isn't any gaze data.
                    #TODO: 502 block 2 does not have much gaze data.
                    if gaze_trial_df.empty:
                        print("No Gaze Data for trial ", trialcode)
                        print("Trial not added to master ", trialcode)
                    else:    
                        #HACK for when there is more steering data than gaze data.
                        mint_gaze = min(gaze_trial_df['viz_timestamp'].values)
                        if mint_gaze > mint:
                            trialdata = trialdata.loc[trialdata['currtime'] > mint_gaze, :]

                        maxt_gaze = max(gaze_trial_df['viz_timestamp'].values)
                        if maxt_gaze < maxt:
                            trialdata = trialdata.loc[trialdata['currtime'] < maxt_gaze, :]


                        print("Gaze Data length: ", len(gaze_trial_df))                                                                                                
                    #   
                        print("Steering Data length: ", len(trialdata))
                        
                        df_stitch = StitchGazeAndSteering(gaze_trial_df,trialdata, latency) #interpolates gaze with simulator timestamps.                        
                        
                        #print("stitch length", len(df_stitch.index))
                        df_stitch = add_all_coord_systems(df_stitch) #adds two new columns: hangle and vangle                                

                        df_stitch = GazeinWorld(df_stitch, midline, trackorigin) #locates gaze in world using converted matlab code. 

                        df_stitch = GazeonPath_inworld(df_stitch) #adds path distance in metres.

                        #df_stitch = GazeonPath_angles(df_stitch) #adds path distance in angles.
                        
                        #pick target positions
                        #pprint(targets)
                        condition = trialdata['condition'].values[0]
                        print("condition", condition)   
                        target_centres = targets.loc[targets['condition']==int(condition),:]
                        #pprint(target_centres)

                        target_centres = target_centres.reset_index(drop=True)

                        #pick starting position.
                        start_x = np.sign(trialdata['posx']).values[0]
                        #print("start pos", trialdata['posx'].values[0])

                        #select targets with opposite sign for xcentre, these will be the ones encountered in that trial
                        target_centres = target_centres.loc[np.sign(target_centres['xcentre'])!=start_x,:]

                        #pprint(target_centres)   
                
                        target_circles = dp.target_position_circles(target_centres)
                        

                        df_stitch = GazeToObjects(df_stitch, target_centres, target_circles)


                        #move code from tidy_data_for_analysis.R to here.
                        #adding some useful columns.
                        df_stitch = mirror_and_add_roadsection(df_stitch)
                                                                        

                        print("stitch length", len(df_stitch.index))
                                
                        #master_gaze = pd.concat([master_gaze,df])
                    
                        #master_stitch = pd.concat([master_stitch,df_stitch])

                        print("APPENDING TRIAL")        
                        #master_segment.to_csv(savedir + "\\SegmentationData_longFormat_rerun_290719.csv")
                        #master_stitch.to_csv(savedir + "\\GazeAndSteering_longFormat_rerun_290719.csv")
                        
                        with open(stitchfilepath, 'a', newline = '') as stitchfile:
                            df_stitch.to_csv(stitchfile, mode='a', header=stitchfile.tell()==0)
                
                except Exception as e:
                    print("CANNOT PROCESS TRIAL:", trialcode)
                    print(e)
                    pass
                
                
            compute_time = timer()-begin
            print("Processing block took %f seconds" % compute_time)
            
    
    #now you've built the master data of all trials, save it.
#    master_gaze.to_csv(savedir + "FramebyFrameGaze_longFormat_100918.csv")
            
            #master_stitch.to_csv(savedir + outfile)


if __name__ == '__main__':

    gazefolderdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/"
    gazefile = 'pupil_data'                
    steerfile = "D:/Trout18_SteeringData_Pooled"
    
    outfile = "/trout18_gazeandsteering_fake_.csv"
        #savedir = "C:/git_repos/sample_gaze_Trout/"
    savedir = "C:/git_repos/Trout18_Analysis/Data"
    main(gazefolderdir, steerfile, savedir, outfile, rerun = False, date = '2019-12-05')
