

"""
script to check influence of marker and surface measurement error on projection through the world.

"""
if __name__ == '__main__':

EH = 1.2

x_norm = np.linspace(.5, .75, 20)
y_norm = np.linspace(.3, .75, 20)

#H, V = np.meshgrid(H, V)

#need to locate on surface, with different widths measured.
def locate_on_surface(x_norm, y_norm):
    
    #return matrix using different width and height measurements.
    pass


#project through to world
def surf_to_world():
    pass


#plot.

hrad = (H*np.pi)/180
vrad = (V*np.pi)/180
    
zground = -EH/np.tan(vrad) 
xground = np.tan(hrad) * zground
lookahead = np.sqrt((xground**2)+(zground**2)) #lookahead distance from midline, before any rotation. Will be in metres.
    
        
        
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
        screen_meas = 196.5, 115.5 #these need checked. measured at 198.5, 112 on 29/07/19
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
    

    def GazeinWorld(df, midline, trackorigin):
    
    #convert matlab script to python to calculate gaze from centre of the road, using Gaze Angles.
    
    def GazeMetrics(row):
        
        EH = 1.2
    
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

        return (lookahead, gazebias, xpog, zpog)
    
    df['lookahead'], df['gazebias'], df['xpog'],df['zpog'] = zip(*df.apply(GazeMetrics,axis=1))    