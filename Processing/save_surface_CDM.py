###script to view and almagamate eye tracking files.
import numpy as np
import sys, os
from file_methods import *
import pickle
import matplotlib.pyplot as plt
import cv2
import csv

"""
Useful docs:
    
To understand marker transform:    
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

Marker centroids & vertices appear to be given in the camera's resolution.

Steps:
    1) Plot markers and gaze position per frame to explore data.
    2) Save in csv file with experiment and trial ID.
    3) Use matlab script -- converted into python -- to generate measures, such as angles etc. (or could do that in this script)
    
    #Screen co-ords.
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+
    
Now I have the markers positions I need to do more with them than simple translate them into screen co-ords.
I also need to correct for tilt of the camera, and correct gaze position in relation to this. 
Look at how their screen tracker and surface transformations do this. 
    
"""

if __name__ == '__main__':
	#rootdir = sys.argv[1] 
    #rootdir = "E:/Masters_17-18_RunningTotal_Lastupdate_250118/PG_AW15F" #directory for eyetracking files. 
   # rootdir = "E:/Masters_17-18_RunningTotal_Lastupdate_250118/KH12F/Crow17_KH12F/001"
    rootdir = "E:/Masters_17-18_RunningTotal_Lastupdate_250118/"
    #save the pupil-corrected files in the analysis folder.
    savedir = "C:/Users/psccmo/OneDrive - University of Leeds/Research/Student-Projects_2017-18/Sparrow-Crow-17/Data/EyetrackingData/"
    resave = False #boolean whether to move files to savedir
    resx, resy = 1280,720
    marker_corners_norm = np.array(((0,0),(1,0),(1,1),(0,1)),dtype=np.float32)
    marker_box = np.array([[[0,0],[1,0],[1,1],[0,1],[0,0]]],dtype=np.float32) #this is for perspectiveTransform.
    
    print (rootdir)
    for dirs in os.walk(rootdir):
        path = str(dirs[0]) + "/"
        print (path)
        if os.path.exists(path + 'pupil_data'):
       # if os.path.exists(path + 'markers.npy'):
            #marker is an 1 to N frames array. 
            #start plot.
            
#            all_markers = np.load(path + 'markers.npy')
#            for frm in all_markers: #each frame
#                t = frm["ts"] #timestamp
#                markers = frm["markers"] #produces a list.
#                m_count = len(markers) #amount of markers detected in that frame.
#                plt.plot()
#                #plt.axis([0,1,0,1])
#                plt.axis([0,resx,0,resy])
#                i = 1
#                for m in markers:
#                   #simply scaling markers doesn't seem to work.
#                    #centre = m['centroid'] #pick centroid for marker.
#                   # c_scaled = (centre[0]/Resolution[0],centre[1]/Resolution[1])  
#                    print ("markers:", i)
#                    i +=1
#                    verts = m['verts']                   
#                    v = np.array(verts, dtype=np.float32) #need to be in this type for perspective transform.
#                    ptrans = cv2.getPerspectiveTransform(marker_corners_norm, v)
#                    pbox =  cv2.perspectiveTransform(marker_box, ptrans)
#                    #pbox = verts
#                    vi=1
#                    for pp in pbox:
#                        for p in pp:
#                            print ("verts:", vi)
#                            vi += 1
#                            #p_scaled = p[0]/resx, p[1]/resy                                                                      
#                            #plt.plot(p_scaled[0],p_scaled[1],'b.')
#                            plt.plot(p[0],p[1],'b.')
#                plt.show()
#                    
                    
                
            #centroids = [m['centroid']for m in markers]
			#print ("Markers: ", markers)
            data = load_object(path + "/pupil_data")
			#data = pickle.load(path + "/pupil_data_corrected")
            #raw data is dictionary with four lists: pupil_positions, gaze_positions, notifications, surfaces.
            #get trialtype
            notes = data["notifications"]
            mynote = notes[-1] #annotation is always at the end. 
            trialtype = mynote['label']
            print('trialtype: ', trialtype)
            
            surfaces = data["surfaces"]
            notes = data["notifications"]
            with open(os.path.join(path, 'gaze_positions_on_surface_uncorrected.csv'), 'w', encoding='utf-8', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                csv_writer.writerow(('world_timestamp', 'surface_timestamp', 'x_norm',
                                    'y_norm', 'on_srf', 'confidence','trialtype'))
                for s in surfaces:
                    #surface has name, uid, m_to_screen, m_from_screen, gaze_on_srf, timestamp, camera_pose_3d
                    #this is the uncorrected surface information obtained online from the surface tracker plugin.
                    ts = s["timestamp"] #world timestamp.
                    gos = s["gaze_on_srf"]
                    if gos is not None:                        
                        for i in range(0,len(gos)):
                            gp = gos[i]
                            csv_writer.writerow((ts,gp['base_data']['timestamp'],
                            gp['norm_pos'][0], gp['norm_pos'][1],                            
                            gp['on_srf'], gp['confidence'], trialtype))
                
            #NOW SAVE INTO ONEDRIVE FOLDER.
            
            #for d in data:
            #    print (d)
			#note = data["notifications"]
			#annotate = note[-1] #annotation is always at the end. 
			#print (annotate)
			#label = annotate['label']
			#print (label)
#			for n in note:
#				print(n)
			
			#gaze = data["gaze_positions"]
			#for g in gaze:
		#		print (g['norm_pos'])
			
           # pupil = data["pupil_positions"]
            #for p in pupil:
             #   print (p)


			#A = load_object(path + "/annotations")
			#for a in A:
			#	label = a['label']
			#	print (label )
			#print (A)
            
            if resave:
                #in string, find the end of the master folder, then take that and the rest and append it to savedir.
                idx = path.find('18/')
                savepath = path[idx+2:]
                savefile = savedir+savepath  
                os.makedirs(savefile)
                save_object(data, savefile + "/pupil_data_corrected")
                np.save(savefile+"/markers.npy",markers)


