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


def correlateTimestamps(df):
   
    #TODO: use the eye_id to average over duplicate timestamps with multiple eyes. 
    #FIXME: This function is slow and doesn't work properly. Dealing with dataframes in this way seems slow. 
    
    #receives panda dataframe and crunches the multiple timestamps.
    world_ts = np.unique(df['world_timestamp'])
    data_by_ts = pd.DataFrame()    
        
    
    def avgAcrossTs(stamps):
        #receives a list of entries associated with that timestamp.
        #take the most confidence. If all equal, average.
        print(stamps)
        mx = stamps['confidence'].max() 
        gp= stamps.loc[stamps['confidence']==mx]
        avggp = gp.mean(0)        
        return avggp
    
    #loop through surface_timestamps and find closest frame index 
    
    for row in range(df.shape[0]):
        #print(row)
        datum = df.iloc[[row]] #needs list selection for dataframe.   
        surf_ts = df.iloc[row,1] #surface timestamp        
        if surf_ts >0 and surf_ts<20:    #check ts is during trial. If not don't process datum.
            tsdiff = world_ts - surf_ts 
            idx = np.argmin(abs(tsdiff))
            datum['frame_index'] = idx #Add new column.
            data_by_ts = pd.concat([data_by_ts,datum])
                
    #now you've correlated timestamps. Return average for duplicate entries. 
    #
    df_new = data_by_ts.groupby(['frame_index']).apply(avgAcrossTs)
   
#    for frame in range(len(world_ts)):
#        
#        #find all the entries.
#        ts_entries = []
#        for i in range(len(data_by_ts)):
#            dt = data_by_ts[i]
#            if dt['frame_index'] == frame:
#                ts_entries.append(i)
#                
#        if len(ts_entries) > 0:
#            Tlist = [data_by_ts[ts_entries[i]] for i in range(len(ts_entries))]    
#            Tdf = pd.DataFrame(Tlist)          
#            d= avgAcrossTs(Tdf)
#            df_new.append(d)            
    
    return df_new
    
def GazeAngles(df):
    
        
    def SurfaceToGazeAngle(gp):
        
        
        #proj_width = 1.965
        #proj_height = 1.115    
        
        #pixel size of markers of total white border marker is 118 x 107. But I think surface is up to black marker edge.
        #measurements of black marker edge in inkscape are ~75 x 72 pixels. 
        #NEED REAL_WORLD MEASUREMENTS OF SURFACE SIZE UP TO BLACK SQUARE.
        #AND HORIZON RELATIVE TO BOTTOM AND TOP OF SQUARE.
        #put the below into a function and call apply: 

        """
        Since the horizon placement is relative I only need to know the marker placements on the screen rather than the extents. 

        in eyetrike_calibration_standard, code used to determine the marker size is:
        self.boxsize = [.8,.5] #xy box size
		self.lowerleft = [.1,.1] #starting corner

        """
        #find half marker size in screen coords. 
        #marker vertical is measured at 7.45 cm on 29/08/19
        #screen_res = 1920, 1080
        boxsize = [.8, .5]
        lowerleft = [.1,.1]
        screen_meas = 198.5, 115.5 #these need checked. measured at 198.5, 112 on 29/07/19
        marker_onscreen_meas = 7.45
        marker_norm_size = (marker_onscreen_meas / screen_meas[1])

        #determine the relative position of .5
        bottom_edge = lowerleft[1] - (marker_norm_size/2)
        top_edge = lowerleft[1] + boxsize[1] + (marker_norm_size/2)

        #horizon at .5
        horizon = .5
        horizon_in_surface = (horizon - bottom_edge) / (top_edge - bottom_edge)

        #print ("horizon_in_surface", horizon_in_surface)

        """        
        #need to check that surfaces is from edges, as in vizard the placement is the bottom left corner
        width = 1.656 #measured at 165.6 cm on 14/12/18 #real-world size of surface, in m.
        height = .634 #measured at 63.4 cm on 18/12/18
        #this should match the defined surface.
        
        centrex = .5 #horiz centre of surface is centre of experiment display.
        
        Horizon_relativeToSurfaceBottom = .455 #Horizon measured at 45.5 above bottom marker value . 45.5/63.5 = .7063

        #it is very important centrey is accurate. make sure you measure up to the true horizon, not the clipped horizon because this was overestimate how far people are looking
        #Measured at 46cm above the bottom marker. 46/60 is .7666667

        
        centrey = Horizon_relativeToSurfaceBottom / height #.7667 #placement of horizon in normalised surface units. Minus off norm_y to get gaze relative to horizon.

        #TODO: CHECK HORIZON MEASUREMENT AS SURFACE DOESN'T ALIGN WITH TOP OF MARKERS. NEED TO CORRECT FOR DISTORTION.
        screen_dist = 1.0 #in metres
        """
        width = 1.656 #measured at 165.6 cm on 14/12/18 #real-world size of surface, in m.
        height = .634 #measured at 63.4 cm on 18/12/18

        #on 290719, measured at 1.65 and 63. These don't match up to the estimated widths given the other parameters, but let's use them for now.
        width = 1.65 #measured at 165.6 cm on 14/12/18 #real-world size of surface, in m.
        height = .63 #measured at 63.4 cm on 18/12/18
        screen_dist = 1.0 #in metres

        """
        #check. width should match up to:
        marker_norm_size_width = (marker_onscreen_meas / screen_meas[0])
        surface_width_norm = boxsize[1] + marker_norm_size_width_width
        estimated_width = surface_width_norm * screen_meas[0]

        surface_height_norm = top_edge - bottom_edge
        estimated_height = surface_height_norm * screen_meas[1]
        """

        centrex = .5
        centrey = horizon_in_surface

        #convert the scale to real-distances from centre.
        x = gp['x_norm']
        y = gp['y_norm']
        real_h = (x-centrex)*width
        real_v = (y-centrey)*height
#	
    	#calculate gaze angle
        hrad = mt.atan(real_h/screen_dist)
        vrad = mt.atan(real_v/screen_dist)
#	
    #	#convert to degrees
        hang = (hrad*180)/mt.pi
        vang= (vrad*180)/mt.pi
#	
        return (hang, vang) 
    
    
    df['hangle'], df['vangle'] = zip(*df.apply(SurfaceToGazeAngle,axis=1))
       
    return df	
    
def LoadSteeringBlock(steerdir, block, pcode):
    #return steering data for trial. 
   # df_steer = pd.DataFrame() #trial frame by frame data
    steering_block = pd.DataFrame() #master data for gaze and steering          

    file_prefix = '_'.join([block, pcode])
    #EXPBLOCK_PP_drivingmode_condition_count
    for fn in glob(steerdir + "/" + file_prefix + "*.csv"):
        
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
        
        steering_block = pd.concat([steering_block,trial_data])                        
    
        
    return(steering_block)
    
def StitchGazeAndSteering(df_gaze, df_steer, latency):
    
    #linearly interpolate so that timestamps match exactly.
    #recv_timestamp should be viz.tick()

    #interpolate the normed positions
    yinterpolater_ynorm = interp1d(df_gaze['recv_timestamp'].values - latency,df_gaze['y_norm'].values)
    y_norm = yinterpolater_ynorm(df_steer['currtime'].values)
                    
    xinterpolater_xnorm = interp1d(df_gaze['recv_timestamp'].values - latency,df_gaze['x_norm'].values)
    x_norm = xinterpolater_xnorm(df_steer['currtime'].values)
    
    df_steer.loc[:,'y_norm'] = y_norm
    df_steer.loc[:,'x_norm'] = x_norm  


    #interpolate confidence values.
    confidenceinterpolater = interp1d(df_gaze['recv_timestamp'].values - latency,df_gaze['confidence'].values)
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

def GazeonPath(df):
    
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
    
    df['gazedistance'], df['on_srf'] = zip(*df.apply(CalculateGazeDistance,axis=1))    
    #row = df.loc[2,:]
    #lookahead, gazebias, xpog, zpog = GazeMetrics(row)
   
    
    return (df)


def TrackMaker(sectionsize):
	
	"""adds oval track with double straight. Returns 4 variables: midline, origin, section breaks, track details"""
	#at the moment each straight or bend is a separate section. So we can alter the colour if needed. But more efficient to just create one line per edge.
#    code copied from vizard trackmaker.

	"""
       ________
	 /   _B__   \ 
	/	/    \   \ 
	| A |    | C |
   _|   |    |   |_
   _| H |    | D |_
	|	|	 |   |
	| G	|	 | E |
	\	\___ /   /
     \ ____F___ /


	A = Empty Straight 
	B = Constant curvature Bend
	C = Straight with Targets.
	D = Interp period (Length = StraightLength / 4.0)
	E = Empty Straight
	F = Constant curvature bend
	G = Straight with Targets
	H = Interp period (Length = StraightLength / 4.0)

	TrackOrigin, centre of track = 0,0. Will be half-way in the interp period.

	"""



	#Start at beginning of 1st straight.
	StraightLength = 40.0 #in metres. 
	InterpProportion = 1.0 #Length of interpolation section relative to the straight sections
	InterpLength = StraightLength * InterpProportion
	InterpHalf = InterpLength / 2.0
	BendRadius = 25.0 #in metres, constant curvature bend.
	SectionSize = sectionsize
	roadwidth = 3.0/2.0
	right_array = np.linspace(np.pi, 0.0, SectionSize) 
	left_array= np.linspace(0.0, np.pi,SectionSize)
	
	
	#trackorigin = [BendRadius, StraightLength/2.0] #origin of track for bias calculation
	trackorigin = [0.0, 0.0]
	trackparams = [BendRadius, StraightLength, InterpLength, SectionSize, InterpProportion]

	#For readability set key course markers. Use diagram for reference
	LeftStraight_x = -BendRadius
	RightStraight_x = BendRadius
	Top_Interp_z = InterpHalf
	Top_Straight_z = InterpHalf+StraightLength
	Bottom_Interp_z = -InterpHalf
	Bottom_Straight_z = -InterpHalf-StraightLength
	
	###create unbroken midline. 1000 points in each section.	
	#at the moment this is a line so I can see the effects. But this should eventually be an invisible array.
	#straight	
	#The interp periods have index numbers of sectionsize / 4. So midline size = SectionSize * 7 (6 sections + two interps)
	midlineSize = SectionSize* (6 + 2 * InterpProportion)
	midline = np.zeros((int(midlineSize),2))

	SectionBreaks = []
	SectionBreaks.append(0)
	SectionBreaks.append(int(SectionSize)) #end of StraightA #1
	SectionBreaks.append(int(SectionSize*2)) #end of BendB #2
	SectionBreaks.append(int(SectionSize*3)) #end of StraightC #3
	SectionBreaks.append(int(SectionSize* (3 + InterpProportion))) #end of InterpD #4
	SectionBreaks.append(int(SectionSize*(4 + InterpProportion))) #end of StraightE #5
	SectionBreaks.append(int(SectionSize*(5 + InterpProportion))) #end of BendF #6
	SectionBreaks.append(int(SectionSize*(6 + InterpProportion))) #end of StraightG #7
	SectionBreaks.append(int(SectionSize*(6 + 2*InterpProportion))) #end of InterpH #8

	#Straight A
	StraightA_z = np.linspace(Top_Interp_z, Top_Straight_z, SectionSize)
	midline[SectionBreaks[0]:SectionBreaks[1],0] = LeftStraight_x
	midline[SectionBreaks[0]:SectionBreaks[1],1] = StraightA_z

	#print (SectionBreaks)
	#print (midline[SectionBreaks[0]:SectionBreaks[1],:])
		
	#Bend B
	i=0
	while i < SectionSize:
		x = (BendRadius*np.cos(right_array[i])) #+ BendRadius 
		z = (BendRadius*np.sin(right_array[i])) + (Top_Straight_z)
		midline[i+SectionBreaks[1],0] = x
		midline[i+SectionBreaks[1],1] = z
		#viz.vertex(x,.1,z)
		#viz.vertexcolor(viz.WHITE)
		xend = x
		i += 1
	
	#StraightC
	rev_straight = StraightA_z[::-1] #reverse
	midline[SectionBreaks[2]:SectionBreaks[3],0] = xend
	midline[SectionBreaks[2]:SectionBreaks[3],1] = rev_straight
	
#		
# 	#InterpD
	InterpD_z = np.linspace(Top_Interp_z, Bottom_Interp_z, int(SectionSize*InterpProportion))
	midline[SectionBreaks[3]:SectionBreaks[4],0] = xend
	midline[SectionBreaks[3]:SectionBreaks[4],1] = InterpD_z

	#StraightE
	StraightE_z = np.linspace(Bottom_Interp_z, Bottom_Straight_z, SectionSize)
	midline[SectionBreaks[4]:SectionBreaks[5],0] = xend
	midline[SectionBreaks[4]:SectionBreaks[5],1] = StraightE_z

	#BendF
	i=0
	while i < SectionSize:
		x = (BendRadius*np.cos(left_array[i]))
		z = -(BendRadius*np.sin(left_array[i])) + (Bottom_Straight_z)
		midline[i+(SectionBreaks[5]),0] = x
		midline[i+(SectionBreaks[5]),1] = z
	#	viz.vertex(x,.1,z)
	#	viz.vertexcolor(viz.WHITE)
		xend = x
		i += 1
	
	#StraightG
	StraightG_z = np.linspace(Bottom_Straight_z, Bottom_Interp_z, SectionSize)
	midline[SectionBreaks[6]:SectionBreaks[7],0] = xend
	midline[SectionBreaks[6]:SectionBreaks[7],1] = StraightG_z

	#InterpG
	InterpG_z = np.linspace(Bottom_Interp_z, Top_Interp_z, int(SectionSize*InterpProportion))
	midline[SectionBreaks[7]:SectionBreaks[8],0] = xend
	midline[SectionBreaks[7]:SectionBreaks[8],1] = InterpG_z

	TrackData = []
	TrackData.append(midline)
	TrackData.append(trackorigin)
	TrackData.append(SectionBreaks)
	TrackData.append(trackparams)

	return TrackData

def main(gazedir, steerdir, savedir, outfile, latency):
	#rootdir = sys.argv[1] 
    #rootdir = "E:\\Trout_rerun_gazevideos"    
    #rootdir = "E:\\Trout_rerun_gazevideos"   
    rootdir = gazedir
    
    #savedir = "E:/Trout_rerun_processed_gazesteeringdata"
    
    
    resave = False #boolean whether to move files to savedir
    #steerdir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Master_SampleParticipants/SteeringData/"
    
   # marker_corners_norm = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
   # marker_box = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32) #this is for perspectiveTransform.
   
    INCLUDE_GAZE = False

    latency = .175 #estimated lag #visual inspection for check_sync_by_plotting.py
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
    
    
        
    #For pilot scripts 
    #STEERING_TIME_CORRECTION = 74.60509197 #Take the first timestamp of the first steering trial, since vizard timestamp isn't zeroed after calibration but pupil timestamp is.
    #EYETRACKING_TIME_CORRECTION = 6568.631288

    gazedata_filename = '/gaze_on_surface.csv'
    #print(pfolder)
    #print ("here")
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) 
        print (path)
        folder_skip = "2B_reviewed"
        if folder_skip in path: continue
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
            ppfolder = path.parts[-2]
#           
            print("ppfolder", ppfolder)            
            expname, pcode, block = ppfolder.split('_')
            print(expname)
            print(pcode)
            print(block)

            #load the steering block

            steering_block_df = LoadSteeringBlock(steerdir, block, pcode)                                                         
            
                #now we have the steering_block_df we want to loop through the trials and add the gaze data the corresponds with the timezones for each trial. We do not need to save the calibration data here.            
            try:
                trialcodes = steering_block_df['trialcode'].unique().tolist()
                
                
                for trial in trialcodes:                            
                    
                    try: 
                        print("Processing: ", trial)                    
                        
                        steering_trial_df = steering_block_df.loc[steering_block_df['trialcode']==trial]

                        #get the timestamps for that particular trial.
                        mint = steering_trial_df['currtime'].values[0] #could also use min and max
                        maxt = steering_trial_df['currtime'].values[-1]

                        #pick the corresponding gaze timestamps
                        lower = gaze_block_df['recv_timestamp'] > (mint - .5) #add half a second buffer so the steering timestamps are within the interpolation rangte.
                        upper = gaze_block_df['recv_timestamp'] < (maxt + .5)
                        gaze_trial_df = gaze_block_df.loc[lower & upper, :].copy() 

                        #HACK for when there isn't any gaze data.
                        #TODO: 502 block 2 does not have much gaze data.
                        if gaze_trial_df.empty:
                            print("No Gaze Data for trial ", trial)
                            print("Trial not added to master ", trial)
                        else:    
                            #HACK for when there is more steering data than gaze data.
                            mint_gaze = min(gaze_trial_df['recv_timestamp'].values)
                            if mint_gaze > mint:
                                steering_trial_df = steering_trial_df.loc[steering_trial_df['currtime'] > mint_gaze, :]

                            maxt_gaze = max(gaze_trial_df['recv_timestamp'].values)
                            if maxt_gaze < maxt:
                                steering_trial_df = steering_trial_df.loc[steering_trial_df['currtime'] < maxt_gaze, :]
                        
                            print("Gaze Data length: ", len(gaze_trial_df))                                                                                                
                        #   
                            print("Steering Data length: ", len(steering_trial_df))
                            
                            df_stitch = StitchGazeAndSteering(gaze_trial_df,steering_trial_df, latency) #interpolates gaze with simulator timestamps.                        

                            df_stitch = GazeAngles(df_stitch) #adds two new columns: hangle and vangle            

                            df_stitch = GazeinWorld(df_stitch, midline, trackorigin) #locates gaze in world using converted matlab code. 

                            df_stitch = GazeonPath(df_stitch) #adds Path Distance and Path Proportion to dataframe.
                            
                            #print (list(df_stitch.columns.values))
                                
                            ###here we add the segmentation, using jami's algorithm.                    
                            ##use interpolated timestamp to match simulator values.
                            ##for some reason this doesn't work as well with the simulator vlaues
                #                                        
                            v = df_stitch['vangle'].values
                            h= df_stitch['hangle'].values
                            t = df_stitch['currtime'].values
                            eye = np.vstack((v,h)).T
                            print('classifying gaze')
                            sample_class, segmentation, seg_class = classify_gaze(t, eye)
                            #sample_class is an array of same dimensions as inputs to classify_gaze, so can be added on to original dataframe.
                            df_stitch['sample_class'] = sample_class                    
                        
                #                        seg_class is an array of dimensions the same as number of segmentations
                #                        segmentation is an nslr.slow class, with segments that have t and x. 
                #                        t is a 2dim array with start and end points. x is a 2x2 array vangle in x[:,0] and hangle in [x:,1]
                #                        plt.plot(t,h,'.')
                #                    
                #                        need to save segment in own file as it has different dimensions.
                            seg_trial = pd.DataFrame()
                            #add segmentation class and identification variables        
                            #         #add some more useful identifiers that may not be in there already
                            
                            block, ppid, sectiontype, condition, count =  trial.split("_") 
                            

                            seg_trial['seg_class'] = seg_class                        
                            seg_trial['ID'] = ppid
                            seg_trial['trialcode'] = trial
                            seg_trial['condition'] = condition
                            seg_trial['count'] = count
                            seg_trial['sectiontype'] = sectiontype
                            seg_trial['sectionorder'] = df_stitch['sectionorder'].values[0]
                            seg_trial['block'] = block
                        
                            for i, segment in enumerate(segmentation.segments):                                                
                                t = np.array(segment.t) # Start and end times of the segment
                                x = np.array(segment.x) # Start and end points of the segment
                                seg_trial.loc[i,'t1'] = t[0]
                                seg_trial.loc[i,'t2'] = t[1]
                                seg_trial.loc[i,'v1'] = x[0,0]
                                seg_trial.loc[i,'v2'] = x[1,0]
                                seg_trial.loc[i,'h1'] = x[0,1]
                                seg_trial.loc[i,'h2'] = x[1,1]
                                                    
                                #here calculate average yaw-rate for segment for the corresponding time period.
                                
                                start = df_stitch['currtime'] >= t[0]
                                end = df_stitch['currtime'] <= t[1]
                                
                                yawrates = df_stitch.loc[start&end,'yawrate']
                                seg_trial.loc[i,'yawrate'] = yawrates.mean()
                                #  plt.plot(t, x[:,1], 'ro-', alpha=0.5)
                                    
                            
                            #plt.show()
                            master_segment = pd.concat([master_segment,seg_trial])
                        
                            print ("added to master df")
                            #master_gaze = pd.concat([master_gaze,df])
                        
                            master_stitch = pd.concat([master_stitch,df_stitch])
                    except Exception as e:
                        print("CANNOT PROCESS TRIAL:", trial)
                        print(e)
                        pass

            except Exception as e:
                print("CANNOT PROCESS FOLDER", ppfolder)
                print(e)
                pass
                
            compute_time = timer()-begin
            print("Processing block took %f seconds" % compute_time)
            
    
    #now you've built the master data of all trials, save it.
#    master_gaze.to_csv(savedir + "FramebyFrameGaze_longFormat_100918.csv")
    print("SAVING MASTER DATAFRAME")        
    #master_segment.to_csv(savedir + "\\SegmentationData_longFormat_rerun_290719.csv")
    #master_stitch.to_csv(savedir + "\\GazeAndSteering_longFormat_rerun_290719.csv")
    master_stitch.to_csv(savedir + outfile)


if __name__ == '__main__':

    gazedir = "C:/git_repos/sample_gaze_Trout" 
    steerdir = "E:/Trout_rerun_steering_pooled"
    savedir = "C:/git_repos/sample_gaze_Trout"
    outfile = "/GazeAndSteering_newlatency.csv"
    main(gazedir, steerdir, savedir, outfile)
