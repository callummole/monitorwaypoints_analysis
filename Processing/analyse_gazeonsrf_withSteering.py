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
        TODO: Since I know the positions of the markers on the screen, and should know their extents, I can calculate the horizon exactly on the assumption that it is at y = .5 in the main screen

        I can get the marker size from real-world marker measurements.

        I can also calculate the ratio by estimating the scaling according to the programmed values.


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
        
        #convert the scale to real-distances from centre.
        x = gp['x_norm']
        y = gp['y_norm']
        real_h = (x-centrex)*width
        real_v = (y-centrey)*height
#	
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
    
def LoadSteering(path, trial, maxt, mint):
    #return steering data for trial. 
   # df_steer = pd.DataFrame() #trial frame by frame data
                        
    filename = path + str(trial) + '.csv' #formula for reading in steering file. 
    print("Steering Filename: ", filename)
    untrimmed_df_steer= pd.read_csv(filename) #read steering data                       
    #df_steer= pd.read_csv(filename) #read steering data                       
#    endtrial = len(importMat) #0 to 1 signifies end of trial.                    
#    trialdata = importMat.iloc[1:endtrial-1,:] #take all of the data, minus the first and last frame.
    
    #print(untrimmed_df_steer) #Not sure why I trimmed this.

    df_steer = untrimmed_df_steer.loc[:,['ID','Age','Gender','Vision','LicenseMonths','frameidx','trackindex','currtime','SWA','posx','posz','yaw','yaw_d','yawrate','steeringbias','autoflag','sectiontype','sectionorder','sectioncount','trialtype','obstaclecolour','obstacleoffset' ]].copy()
    
    #STEERING_TIME_CORRECTION = df_steer.loc[0,'currtime']
    
   # print("Steering Correction:", STEERING_TIME_CORRECTION)
    
   # df_steer['currtime'] = df_steer['currtime'] #- STEERING_TIME_CORRECTION
    
    #Since stitching together gaze and steering data relies on creating an interpolation function for the gaze data, then passing it the steering timestamps, we need the steering timestamps to be within the range of values for gaze timestamps.
        
    lower = df_steer['currtime']>mint
    upper = df_steer['currtime']<maxt
    
    df_steer = df_steer.loc[lower&upper, :].copy()
        
    return(df_steer)
    
def StitchGazeAndSteering(df, df_steer):
    
    #using gaze angles, linearly interpolate simulator angles. Frame rate is high enough that using the 'fake' gaze data isn't a problem.               
    #use all gaze angles to interpolate
    #This function relies on timestamps of simulator and pupil-labs to be synced at start of each trial.
    yinterpolater = interp1d(df['gaze_timestamp'].values,df['vangle'].values)
    y = yinterpolater(df_steer['currtime'].values)
                    
    xinterpolater = interp1d(df['gaze_timestamp'].values,df['hangle'].values)
    x = xinterpolater(df_steer['currtime'].values)

    df_steer['vangle'] = y
    df_steer['hangle'] = x    


    #Also add normed position on surface.
    yinterpolater_ynorm = interp1d(df['gaze_timestamp'].values,df['y_norm'].values)
    y_norm = yinterpolater_ynorm(df_steer['currtime'].values)
                    
    xinterpolater_xnorm = interp1d(df['gaze_timestamp'].values,df['x_norm'].values)
    x_norm = xinterpolater_xnorm(df_steer['currtime'].values)

    df_steer['y_norm'] = y_norm
    df_steer['x_norm'] = x_norm  

    # #interpolate confidence values.
    # confidenceinterpolater = interp1d(df['gaze_timestamp'].values,df['confidence'].values)
    # confidence = confidenceinterpolater(df_steer['currtime'].values)

    # df_steer['confidence'] = confidence

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


if __name__ == '__main__':
	#rootdir = sys.argv[1] 
    rootdir = "E:\\EyeTrike_Backup\\Recordings\\Trout\\ExperimentProper"    
    savedir = "C:\\Users\\psccmo\\Trout18_Analysis\\"
    resave = False #boolean whether to move files to savedir
    #steerdir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Master_SampleParticipants/SteeringData/"
    steerdir = "D:\\Trout18_SteeringData_Pooled\\"
    resx, resy = 1280,720
   # marker_corners_norm = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
   # marker_box = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32) #this is for perspectiveTransform.
   
    INCLUDE_GAZE = False
    #CREATE MIDLINE. Use Function from Main Experiment
    TrackData = TrackMaker(10000)
    midline = TrackData[0]     
    trackorigin =TrackData[1]
    
    #varnames = np.array('pp','tn','cn','Fix','Bend','Exp',')
    #measures: LookaheadDistance; HighestFixationDensity,  
    #create a dataframe for averaged measures 
    #MATLAB outtable = cell2table(out,'VariableNames',{'pp', 'tn', 'cn', 'Fix', 'Bend', 'SB','RMS',...
    #'AvgYaw','StdYaw','AvgSWA','StdSWA','AvgSWVel'});
    
    pcodes = range(0,26) #participant code range
    exp = 'Trout'
    ##0 = Attract_Narrow, 1=Attract_Medium, 2=Attract_Wide, 3=Avoid_Narrow, 4=Avoid_Medium, 5=Avoid_Wide
    #cndtcodes = ['Attract_Narrow','Attract_Medium','Attract_Wide','Avoid_Narrow','Avoid_Medium','Avoid_Wide']   
    cndtcodes = ['Attract_Narrow','Attract_Wide','Avoid_Narrow','Avoid_Wide']   
   
   #steering starting time. 125.4422454. Pilot has bug since vizard timer not reset after calibration / accuracy. So minus starting time to reset.

        
#    master_gaze = pd.DataFrame() #master data for gaze file.
    master_segment = pd.DataFrame() #master data for segment file.                
    master_stitch = pd.DataFrame() #master data for gaze and steering            
    
    
        
    #For pilot scripts 
    #STEERING_TIME_CORRECTION = 74.60509197 #Take the first timestamp of the first steering trial, since vizard timestamp isn't zeroed after calibration but pupil timestamp is.
    #EYETRACKING_TIME_CORRECTION = 6568.631288

    gazedata_filename = '\\gaze_on_surface_Corrected.csv'
    #print(pfolder)
    #print ("here")
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) 
        print (path)
        
        if os.path.exists(path +gazedata_filename):                    
            
            """                    
            TROUT PARTICULARS: There is a trial_timestamps file that contains the start and finish times (on the eyetrike)
            of the individual trials. Use this to partial out the main gaze csv file.

            - For each row in eyetrack_timestamp, select the corresponding data within the start and finish range from the large gaze_df.
            - Then add the steering. Then stitch together.
            
            file has columns: 'world_timestamp', 'surface_timestamp', 'x_norm', 'y_norm', 'on_srf', 'confidence'
            There may be more than one estimate for each timestamp, and they may not be in order.
            Where they is more than one, either: Take the higher conf, or take the avg if conf is equal.
            
            STEPS:
            1) Function to convert norm_pos to x&y angular deviation from centre of screen (horizon). DONE
            2) Function to locate angles in-world: Lookahead distance; Gaze relative to centre of road.
            3) Output as csv in trial root folder.
            4) Function to calculate some basic measures dispersal measures from angular deviation. These are average measures per trial
            5) Measure up with condition flags from the annotation at the start of the trial.
            5) Collate measures across participants into long-format for R processing.                     
            """                                                               
            
            allgaze_df = pd.read_csv(path+gazedata_filename, sep=',',header=0)                                        
            
            allgaze_df = GazeAngles(allgaze_df) #adds two new columns: hangle and vangle                     
            
            eyetrike_timestamps_df = pd.read_csv(path+'/Eyetrike_trialtimestamps.csv')
            
            ### #Needs a correction because the 'reset' time takes place before the instructions. ###


            ### SHOULDN'T NEED THE FOLLOWING CORRECTION FOR THE PROPER EXPERIMENT ###

            #EYETRACKING_TIME_CORRECTION = allgaze_df.loc[0,'gaze_timestamp']
            #print("ET Correction:", EYETRACKING_TIME_CORRECTION)
            
            #Find the first steering trialcode. 
            # #first t1 = first trial.
            # startt1 = min(eyetrike_timestamps_df['t1'])
            # firsttrial_idx = eyetrike_timestamps_df['t1'] == startt1
            # firsttrial_code = eyetrike_timestamps_df.loc[firsttrial_idx, 'trialcode'].values[0]
            
            # print ("firsttrial", firsttrial_code)
            # filename = steerdir + str(firsttrial_code) + '.csv' #formula for reading in steering file. 
            # #print("Steering Filename: ", filename)
            # firsttrial_steering = pd.read_csv(filename) #read steering data                       
            # #STEERING_TIME_CORRECTION = firsttrial_steering.loc[0,'currtime']
            
            #print ("Steering_time_correction:", STEERING_TIME_CORRECTION)                        
            
            #hack for pilot
            #allgaze_df['gaze_timestamp'] = allgaze_df['gaze_timestamp'] #- EYETRACKING_TIME_CORRECTION 

            ### END OF CORRECTION FOR MISMATCHED STARTING TIMESTAMPS ####

            #loop through eyetrike timestamps.
            for index, row in eyetrike_timestamps_df.iterrows():
                
                begin = timer()
                
                mint = row['t1']#- EYETRACKING_TIME_CORRECTION
                #print (mint)
                maxt = row['t2']# - EYETRACKING_TIME_CORRECTION
            
                #print ("Mint", mint)
                #print ("Maxt", maxt)
                
                trial = row['trialcode']
                
                print("Processing: ", trial)    
                
                #hack for pilot
                #Use Eyetracking timestamps file to select the right portion of the eyetracking datafile. 
                
                lower = allgaze_df['gaze_timestamp'] > mint 
                upper = allgaze_df['gaze_timestamp'] < maxt
                trial_gazedata = allgaze_df.loc[lower & upper, :].copy() 
                
                print("Gaze Data length: ", len(trial_gazedata))
                #carry over section order.
                
                                       
                #### Load Steering Timestamps of Same Trial ###########                        
                #recalculate max and min so that gaze and steering data span same range for interpolation
                mint = min(trial_gazedata['gaze_timestamp'].values)
                maxt = max(trial_gazedata['gaze_timestamp'].values)
                
                df_steer = LoadSteering(steerdir,trial,maxt, mint) #loads steering file into a df.
            #   
                print("Steeriing Data length: ", len(df_steer))
               
                df_stitch = StitchGazeAndSteering(trial_gazedata,df_steer) #interpolates gaze with simulator timestamps.                        
                                
                df_stitch = GazeinWorld(df_stitch, midline, trackorigin) #locates gaze in world using converted matlab code. 

                df_stitch = GazeonPath(df_stitch) #adds Path Distance and Path Proportion to dataframe.
               
                print (list(df_stitch.columns.values))
                
                #Add trial identifiers to df_stitch.

                df_stitch['trialcode'] = trial
                df_stitch['count'] = row['count'] 
                df_stitch['condition'] = row['condition']        
                df_stitch['block'] = row['block']                              
                
                #rename some df_stitch columns.
#                        df_stitch.rename(index=str, columns={"ID": "pp_id", "trialtype": "condition"})
                
#                        print (list(df_stitch.columns.values))
                                         
#                    print("trial_gazedata5")             
                #df_stitch['sectiontype'] = row['sectiontype']
                #df_stitch['sectionorder'] = sectionorder
	
                    
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
                seg_trial['seg_class'] = seg_class                        
                seg_trial['ID'] = row['pp']
                seg_trial['trialcode'] = trial
                seg_trial['condition'] = row['condition']
                seg_trial['count'] = row['count']    
                seg_trial['sectiontype'] = row['sectiontype']
                seg_trial['sectionorder'] = df_stitch['sectionorder'].values[0]
                seg_trial['block'] = row['block']                              
            
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
            
                compute_time = timer()-begin
                print("Classifying gaze took %f seconds" % compute_time)
                
    
    #now you've built the master data of all trials, save it.
#    master_gaze.to_csv(savedir + "FramebyFrameGaze_longFormat_100918.csv")        
    master_segment.to_csv(savedir + "SegmentationData_longFormat_25PPs_181218.csv")
    master_stitch.to_csv(savedir + "GazeAndSteering_longFormat_25PPs_181218.csv")



