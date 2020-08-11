import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import sample

screen_res = 1920, 1080
screen_meas = 1.985, 1.115 #these need checked. measured at 198.5, 112 on 29/07/19
centre = .5, .5 #middle of screen, and horizon.

def surface_to_screen(srf_coords):

    """map surface coordinates to screen coordinates
    
    marker pixel coords (0,0 is top left, 1920, 1080 is bottom right)
    measured in GIMP from a screenshot in August 2019.
    TOP left = 155, 392
    BOTTOM left = 155, 1005 
    BOTTOM right = 1764, 1005
    TOP right = 1764, 392    
    
    """

    top_left_pix = 155, (screen_res[1] - 392)
    bottom_left_pix = 155, (screen_res[1] - 1005)
    bottom_right_pix = 1764, (screen_res[1] - 1005)
    top_right_pix = 1764, (screen_res[1] - 392)

    pix_size = np.subtract(top_right_pix, bottom_left_pix)

   # boxsize_norm_check = np.divide(pix_size, screen_res)
    #print("boxsize_check", boxsize_norm_check)

    screen_coords_pix = np.add(np.multiply(srf_coords, pix_size), bottom_left_pix) #rescale then shift

    #print("surf coords", srf_coords)
    #print("screen_coords_pix", screen_coords_pix)

    screen_coords_norm = np.divide(screen_coords_pix, screen_res)

    #print("screen_coords_norm", screen_coords_norm)

    """
    boxsize_norm = [.8, .5]
    lowerleft_centroid_norm = [.1,.1]
    topleft_centroid_norm = 

    lower_left_centroid_pix = np.multiply(lowerleft_centroid_norm, screen_res)
    top_left_centroid_pix = np.
    
    marker_pix_size = 50, 50 #needs to be accurate.
    marker_norm_size = np.divide(marker_pix_size, screen_res)
    screen_coords = np.add(np.multiply(srf_coords, boxsize), lowerleft_centroid_norm) #rescale then shift.
    """    

    return (screen_coords_norm)

def screen_to_angles(screen_coords):

    """
    maps screen to gaze angle
    
    TODO: check with Jami whether one needs the screen_meas
    
    """
    real_meas = np.multiply(np.subtract(screen_coords, centre), screen_meas)

#    print("screen_coords", screen_coords)

 #   print("real_meas", real_meas)

    #distance away from screen is just 1 m. SO you can do:
    #calculate gaze angle
    gazeangles = np.degrees(np.arctan(real_meas))

 #   print("gaze_angles", gazeangles)
    
    return (gazeangles)

def angles_to_screen(gazeangles):
    
    real_meas = np.tan(np.radians(gazeangles))

    screen_coords = np.add(np.divide(real_meas, screen_meas), centre)

    return(screen_coords)

def angles_to_world(angles, position, yaw):

    EH = 1.2 #matched to simulated eye height in vizard.
    rads = np.radians(angles)
    heading_rads = -np.radians(yaw) #need negative yaw to rotate clockwise.

    zground = -EH/np.tan(rads[1]) 
    xground = np.tan(rads[0]) * zground
    lookahead = np.sqrt((xground**2)+(zground**2)) #lookahead distance from midline, before any rotation. Will be in metres.
    
    #rotate point of gaze using heading angle.
    xrotated = (xground * np.cos(heading_rads)) - (zground * np.sin(heading_rads))
    zrotated = (xground * np.sin(heading_rads)) + (zground * np.cos(heading_rads))

    #add coordinates to current world position.
    xpog = xpos+xrotated
    zpog = zpos+zrotated 

    return(lookahead, xpog, zpog)

def target_position_circles(centres):

    target_arrays = []
   
    t = np.linspace(0, 2*np.pi, 500)
    for index, row in centres.iterrows():

        radius = row['target_radius']
        xcentre = row['xcentre']
        zcentre = row['zcentre']

        x_list = []
        z_list = [] 

        for u in t:
        
            x = xcentre +  radius*np.cos(u)
            x_list.append(x)

            z = zcentre + radius*np.sin(u)
            z_list.append(z)

        target_arrays.append([x_list, z_list])

    return(target_arrays)

def world_to_angles(trajposition, viewpos, viewyaw):

    """given a viewing position and yaw, maps a world position to angular offsets on the screen"""

    #minus the viewing position.
    EH = 1.2

    pos_rel_origin = np.subtract(trajposition, viewpos)
    
    heading_rads = np.radians(viewyaw) #keep yaw positive to rotate counter-clockwise

    pos_unrotated_x = pos_rel_origin[:,0] * np.cos(heading_rads) - pos_rel_origin[:,1] * np.sin(heading_rads)
    pos_unrotated_z = pos_rel_origin[:,0] * np.sin(heading_rads) + pos_rel_origin[:,1] * np.cos(heading_rads)

    h_angle = np.degrees(np.arctan(pos_unrotated_x / pos_unrotated_z))
    v_angle = np.degrees(np.arctan(-EH/pos_unrotated_z))    

    gazeangles = np.transpose(np.array([h_angle, v_angle]))
    return( gazeangles ) 


def select_trial_targets(traj, targets):

    """returns only the targets encountered in that trial"""

    first_pos_x = traj[0,0]
    
    if first_pos_x <0: #if start on left of track.
        targets_idx = [0, 1, 2]
    else:
        targets_idx = [3, 4, 5]

    selected_targets = targets.loc[targets['targetindex'] in targets_idx, :]

    return(selected_targets)

def gaze_to_object():
    pass

def minimum_distance(point, line):

    #print("point", point)
    #print("line", line)
    
    distances = np.subtract(line, point)

    distances = distances[~np.isnan(distances)]
    
    minimum_distance = min(abs(distances))

    return(minimum_distance)

if __name__ == '__main__':
	#rootdir = sys.argv[1] 
    
    #filename = "C:/git_repos/Trout18_Analysis/Data/GazeAndSteering_longFormat_rerun_290719.csv"
    filename = "C:/git_repos/sample_gaze_Trout/GazeAndSteering_newlatency_tidied.csv"
    savepath = "C:/git_repos/Trout18_Analysis/Data"
    #savepath = os.path.split(filename)[0]       

    #PLOT = True

#    PLOT = True
    PLOT = True

    if PLOT:
        #axes limits
        angles_limits_bottom = screen_to_angles([0,0])
        angles_limits_top = screen_to_angles([1,1])

        #track
        track = pd.read_csv(savepath + "/track_with_edges.csv")
        inside_edge = track['insidex'].values, track['insidez'].values
        outside_edge = track['outsidex'].values, track['outsidez'].values

    #load target positions
    targets_all = pd.read_csv(savepath + "/TargetPositions.csv")
     
                                                                   
    steergaze_df = pd.read_csv(filename, sep=',',header=0)                

    output_df = pd.DataFrame()                         

    trialcodes = steergaze_df['trialcode'].unique().tolist()

    picked_trials = sample(trialcodes, 5)

    #picked_trials.insert(0, "1_501_0_0_0")

    dodgy_trials = ["2_203_0_0_1"]


    #for trial in picked_trials:
    for trialcode, trialdata in steergaze_df.groupby('trialcode'):
        if trialcode in dodgy_trials: continue
        print("Processing: ", trialcode)                    
                        
        #trialdata = steergaze_df.loc[steergaze_df['trialcode']==trial].copy()
        #trialdata.reset_index()
        trialdata = trialdata.copy()
        srf_x = trialdata['x_norm'].values
        srf_y = trialdata['y_norm'].values

        screen_coords_norm = surface_to_screen(np.transpose(np.array([srf_x, srf_y])))

        trialdata['screen_x_norm'] = screen_coords_norm[:,0]
        trialdata['screen_y_norm'] = screen_coords_norm[:,1]

    #   plt.plot(range(len(screen_coords_norm)), screen_coords_norm[:,1], 'b-')        
    #   plt.show()

        gaze_angles = screen_to_angles(screen_coords_norm)

        trialdata['hangle_new'] = gaze_angles[:,0]
        trialdata['vangle_new'] = gaze_angles[:,1]

        
        #plot differences in projection from previous.
    #   hangle = trialdata['hangle'].values
    #   vangle = trialdata['vangle'].values

    #   hangle_new, vangle_new = gaze_angles[:,0], gaze_angles[:,1]

        """        
        plt.plot(range(len(hangle)), hangle, 'b-')
        plt.plot(range(len(hangle_new)), hangle_new, 'r-')
        plt.show()

        plt.plot(range(len(hangle)), vangle, 'b-')
        plt.plot(range(len(hangle_new)), vangle_new, 'r-')
        plt.show()
        """

        traj_x = trialdata['posx'].values #use the unmirrored values
        traj_z = trialdata['posz'].values #use the unmirrored values

        trajectory = np.transpose(np.array([traj_x, traj_z]))

        #no need to deal with unencountered targets
        #targets = select_trial_targets(trajectory, targets_all)        
        targets = targets_all   

        #pick target positions
        condition = trialdata['condition'].values[0]
        target_centres = targets.loc[targets['condition']==condition]

        #calculate circular target arrays on ground
        #print("target centres", target_centres)
        target_circles = target_position_circles(target_centres)
        #calc targets

        for index, (_, row) in enumerate(trialdata.iterrows()):

            yaw = row['yaw']
            viewpoint = row['posx'], row['posz']

            print(yaw)
            print(viewpoint)

            gaze_on_screen = row['hangle_new'], row['vangle_new']

            #calc gaze to path
            future_traj = trajectory[index:,:]
            traj_angles = world_to_angles(future_traj, viewpoint, yaw)
            angular_gaze_distance = minimum_distance(gaze_on_screen, traj_angles)

            trialdata.loc[index,'angular_gaze_on_path'] = angular_gaze_distance

            """
            #gaze to object for all three targets 
            gaze_to_object = []
            for tc in target_centres:
                                
                target_centre_angles = world_to_angles(tc, viewpoint, yaw)
                angular_gaze_to_object = minimum_distance(gaze_on_screen, target_centre_angles)                            
                gaze_to_object.append(angular_gaze_to_object)

            min_gaze = min(np.array(gaze_to_object))
            trialdata.loc[index, 'angular_gaze_to_object_1'] = gaze_to_object[0]
            trialdata.loc[index, 'angular_gaze_to_object_2'] = gaze_to_object[1]
            trialdata.loc[index, 'angular_gaze_to_object_3'] = gaze_to_object[2]
            trialdata.loc[index, 'angular_gaze_to_object_minimum'] = min_gaze

            """

            #print("traj_angles", traj_angles )
            #print("traj_shape", traj_angles.shape)                        
            
            traj_h, traj_v = traj_angles[:,0], traj_angles[:,1]
            hang, vang = row['hangle'], row['vangle']

            if PLOT:

                #compute track from viewpoint.
                inside_edge_angles = world_to_angles(np.transpose(inside_edge), viewpoint, yaw)
                outside_edge_angles = world_to_angles(np.transpose(outside_edge), viewpoint, yaw)

                plt.plot(traj_h, traj_v, 'm.', markersize = 2)

                #remove any above the horizon
                inside_edge_angles = inside_edge_angles[inside_edge_angles[:,1]<0,:]
                outside_edge_angles = outside_edge_angles[outside_edge_angles[:,1]<0,:]
                    
                plt.plot(inside_edge_angles[:,0], inside_edge_angles[:,1], 'k.', markersize = .5)
                plt.plot(outside_edge_angles[:,0], outside_edge_angles[:,1], 'k.', markersize = .5)


                #compute target arrays from viewpoint
                for target in target_circles:

                    target_circle = np.squeeze(np.array(target))
                    target_angles = world_to_angles(np.transpose(target_circle), viewpoint, yaw)

                    target_angles = target_angles[target_angles[:,1]<0,:]

                    if condition in [0,1]:
                        mycolour = 'b.'
                    elif condition in [2,3]:
                        mycolour = 'r.'

                    
                    plt.plot(target_angles[:,0], target_angles[:,1], mycolour, markersize = .5)
                                        

                if row['confidence'] > .6: #low confidence produces zero values
                    
                    plt.plot(hang, vang, 'go', markersize = 5)
                else:
                    pass    
                

                plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
                plt.xlim(angles_limits_bottom[0],angles_limits_top[0])
                plt.title(trialcode)
                plt.pause(.016)                    
                plt.cla()
                        
        plt.show()

        output_df = pd.concat([output_df,trialdata])

    #output_df.to_csv(savepath + "/sample_participants_newlatency.csv")

