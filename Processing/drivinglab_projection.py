import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from random import sample

screen_res = 1920, 1080
screen_meas = 1.985, 1.115 #these need checked. measured at 198.5, 112 on 29/07/19

#centre is not quite the middle of the screen due to the cave projection. Calculated later after the functions are defined.
#centre = .5, .5 #middle of screen, and horizon.



# The cave and screen spec. Should matched up to myCave.py to get correct projection.
EH = 1.2
width = 1920
height = 1080
z = 1
x = 1.985
y = 1.115
cavewall_heightfromground = .665
centre_cave_screen = cavewall_heightfromground + (y/2.0)
cave_eh_rotation = np.arctan((EH - centre_cave_screen / z))

"""
marker pixel coords (0,0 is top left, 1920, 1080 is bottom right)
measured in GIMP from a screenshot in August 2019.
TOP left = 155, 393
BOTTOM left = 155, 1005 
BOTTOM right = 1764, 1005
TOP right = 1764, 393

"""
top_left_pix = 155, (screen_res[1] - 393)
bottom_left_pix = 155, (screen_res[1] - 1005)
bottom_right_pix = 1764, (screen_res[1] - 1005)
top_right_pix = 1764, (screen_res[1] - 393)
#print(np.degrees(cave_eh_rotation))

# Functions to compute f from cave dimensions
def cave_to_fov(x, z):
	return 2.0*np.arctan((x/2.0)/z)

def fov_to_f(fov, w):
	return w/(np.tan(fov/2.0)*2.0)

cave_to_f = lambda x, z, w: fov_to_f(cave_to_fov(x, z), w)

# Compute the camera matrix
cx = width/2.0
cy = height/2.0
fx = cave_to_f(x, z, width)
fy = cave_to_f(y, z, height)
M = np.array([
	[fx, 0, cx],
	[0, fy, cy],
	[0,  0, 1]
	])

def cam_to_screen_homo(pos_vec):
	
	"""uses camera projection matrix to map world point to screen. Takes a position vector relative to camera [x_cam, z_cam]"""
	
	pos_vec = np.array(pos_vec)
	
	if pos_vec.shape[0] == 2: #no y
		eh_array = np.repeat(np.array([-EH]), pos_vec.shape[1])
		pos_vec = [pos_vec[0], eh_array, pos_vec[1]]    
	elif pos_vec.shape[0] == 3:
		pass
	else:
		Warning("invalid position vector. shape[0] should be vec of len 2 or 3")
	
	
	
	p_homo = np.dot(M, pos_vec)
	p_screen = p_homo[:2]/p_homo[-1] #perspective divide

	p_screen = np.transpose(np.array(p_screen))

	return(p_screen, p_homo[-1])

def screen_to_cam_homo(p_screen):

	"""uses camera projection matrix to map screen point to sworld. Takes a screen pixel vector on the screen"""

	#TODO: finish up

	p_screen = np.transpose(np.array(p_screen))
	
	pass


def cam_to_screen(pos_vec):

	"""uses good ol' maths, no matrix projection involved. Takes a position vector relative to camera [x_cam, z_cam]"""

	pos_vec = np.array(pos_vec)
	if pos_vec.shape[0] == 2: #no y        
		pos_vec = [pos_vec[0], -EH, pos_vec[1]]
	elif pos_vec.shape[0] == 3:
		pass
	else:
		Warning("invalid position vector. Len 2 or 3.")

	x_cam, y_cam, z_cam = pos_vec

	x_screen = fx*x_cam/z_cam + cx
	y_screen = fy*y_cam/z_cam + cy

	p_screen = np.transpose(np.array([x_screen, y_screen]))

	return(p_screen)


def rotate_rel_camera(worldposition, viewpos, viewyaw):

	"""rotates an in-world point so that the camera is the origin. Only takes X,Z position arguments"""
	worldposition = worldposition.reshape(-1,2)
	pos_rel_origin = np.subtract(worldposition, viewpos)
	
	heading_rads = np.radians(viewyaw) #keep yaw positive to rotate counter-clockwise

	#rotate around yaw
	pos_unrotated_x = pos_rel_origin[:,0] * np.cos(heading_rads) - pos_rel_origin[:,1] * np.sin(heading_rads)
	pos_unrotated_z = pos_rel_origin[:,0] * np.sin(heading_rads) + pos_rel_origin[:,1] * np.cos(heading_rads)

	return([pos_unrotated_x, pos_unrotated_z])

def rotate_rel_camera_cave(worldposition, viewpos, viewyaw):

	"""rotates an in-world point so that the camera is the origin. Only takes X,Z position arguments"""


	"""
	To adjust for the discrepency between the difference in the cave front wall middle and the EH we rotate the pitch.

	"""
	#print("worldpos", worldposition[:,1])
	worldposition = worldposition.reshape(-1,2)
	#viewpos = viewpos.reshape(-1,2)

	pos_rel_origin = np.subtract(worldposition, viewpos)
	
	heading_rads = np.radians(viewyaw) #keep yaw positive to rotate counter-clockwise

	#rotate around yaw
	

	pos_unrotated_x = pos_rel_origin[:,0] * np.cos(heading_rads) - pos_rel_origin[:,1] * np.sin(heading_rads)
	pos_unrotated_z = pos_rel_origin[:,0] * np.sin(heading_rads) + pos_rel_origin[:,1] * np.cos(heading_rads)

	#rotate pitch
	y = -EH
	
	pos_cave_y = y * np.cos(-cave_eh_rotation) - pos_unrotated_z * np.sin(-cave_eh_rotation)
	pos_cave_z = y * np.sin(-cave_eh_rotation) + pos_unrotated_z * np.cos(-cave_eh_rotation)

	pos_cave_x = pos_unrotated_x

	cave_pos = np.array([pos_cave_x, pos_cave_y, pos_cave_z])

	#clip = 100
	#cave_pos[:, pos_unrotated_z>clip] = np.nan 
	#cave_pos[:, pos_unrotated_z<0] = np.nan 

	return(cave_pos)

def surface_to_screen(srf_coords):

	"""map surface coordinates to screen coordinates    
	
	"""


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
	maps screen_normalised to gaze angle
	
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

	#these are normalised
	screen_coords = np.add(np.divide(real_meas, screen_meas), centre)

	return(screen_coords)

def angles_to_world(angles, position = [0,0], yaw = 0):

	EH = 1.2 #matched to simulated eye height in vizard.
	angles = angles.reshape(-1,2)
	rads = np.radians(angles)
	print(rads.shape)
   # heading_rads = -np.radians(yaw) #need negative yaw to rotate clockwise.

	zground = -EH / np.tan(rads[:,1]) #negative vangle = below horizon
	xground = np.tan(rads[:,0]) * zground
	#xground = (-EH * np.tan(rads[:,0])) / np.tan(rads[:,1])
	#lookahead = np.sqrt((xground**2)+(zground**2)) #lookahead distance from midline, before any rotation. Will be in metres.
	
	#rotate point of gaze using heading angle. TODO: not correct.
	#xrotated = (xground * np.cos(heading_rads)) - (zground * np.sin(heading_rads))
	#zrotated = (xground * np.sin(heading_rads)) + (zground * np.cos(heading_rads))
	#print(xrotated)
	#print(zrotated)
	xpog = xground
	zpog = zground

	#plt.plot(xpog,zpog, 'b.')
	#plt.show()

	#add coordinates to current world position.
	#xpog = position[0]+xrotated
	#zpog = position[1]+zrotated 


	return(xpog, zpog)



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

def world_to_angles_through_screen(trajposition, viewpos = [0,0], viewyaw = [0]):

	""" maps angles through screen first so the offset cave projection taking into account"""

	pixel_vec, depth = world_to_screen_homo_cave(trajposition, viewpos, viewyaw)
	screen_norms = pixel_vec / screen_res
	angles = screen_to_angles(screen_norms)

	return(angles, depth)

def angles_to_world_through_screen(angles, viewpos, viewyaw):

	"""maps angles to world through cave projection"""
	#TODO: finish
	screen_norms = angles_to_screen(angles)
	

	pass
	
	


def world_to_angles(trajposition, viewpos = [0,0], viewyaw = [0]):

	"""given a viewing position and yaw, maps a world position to angular offsets on the screen
	
	To be consistent with slightly offset gaze, map onto screen and then calculate angles.
	
	"""

	#minus the viewing position.
	EH = 1.2

	pos_unrotated_x, pos_unrotated_z = rotate_rel_camera(trajposition, viewpos, viewyaw)

	h_angle = np.degrees(np.arctan(pos_unrotated_x / pos_unrotated_z))
	v_angle = np.degrees(np.arctan(-EH/pos_unrotated_z))    

	gazeangles = np.transpose(np.array([h_angle, v_angle]))
	return( gazeangles ) 

def world_to_screen(worldpos, viewpos, viewyaw):

	cam_x, cam_z = rotate_rel_camera(worldpos, viewpos, viewyaw)
	pixel_vecs = cam_to_screen([cam_x, cam_z])
	return(pixel_vecs)

def world_to_screen_cave(worldpos, viewpos, viewyaw):

	cam_x, cam_z = rotate_rel_camera_cave(worldpos, viewpos, viewyaw)
	pixel_vecs = cam_to_screen([cam_x, cam_z])
	return(pixel_vecs)

def world_to_screen_homo(worldpos, viewpos, viewyaw):

	cam_x, cam_z = rotate_rel_camera(worldpos, viewpos, viewyaw)
	pixel_vecs, depth = cam_to_screen_homo([cam_x, cam_z])
	return(pixel_vecs, depth)

def world_to_screen_homo_cave(worldpos, viewpos, viewyaw):

	cam_x, cam_y, cam_z = rotate_rel_camera_cave(worldpos, viewpos, viewyaw)
	pixel_vecs, depth = cam_to_screen_homo([cam_x, cam_y, cam_z])
	return(pixel_vecs, depth)

def screen_to_world_home_cave(screenpos, viewpos, viewyaw):

	#TODO: finish
	if max(screenpos) <= 1:
		screenpos *= screen_res #in pixels
	
	pass

horiz_world_pos = np.transpose(np.array([[0],[100]]))
horiz_viewpoint = 0, 0
horiz_yaw = 0
centre, _ = world_to_screen_homo_cave(horiz_world_pos, horiz_viewpoint, horiz_yaw)
centre /= screen_res

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
	min_idx = np.argmin(abs(distances))

	return(min_idx, minimum_distance)



def plot_perspective_view(viewpoint = (-25, 40), yaw = 0, ax = None):
	"""plots a perspective view viewpoint and yaw, returns fig and axes"""
	
	track = pd.read_csv("../Data/track_with_edges.csv")
	inside_edge = track['insidex'].values, track['insidez'].values
	outside_edge = track['outsidex'].values, track['outsidez'].values

	angles_limits_bottom = screen_to_angles([0,0])[0]

	#print(angles_limits_bottom)
	angles_limits_top = screen_to_angles([1,1])[0]

	pixels_limits_top = [1920,1080]
	
	#compute track from viewpoint.
	inside_edge_angles, zi = world_to_angles_through_screen(np.transpose(inside_edge), viewpoint, yaw)
	outside_edge_angles, zo = world_to_angles_through_screen(np.transpose(outside_edge), viewpoint, yaw)    

	#remove any above the horizon	
	print(zi)	
	inside_edge_angles = inside_edge_angles[zi>0,:]
	outside_edge_angles = outside_edge_angles[zo>0,:]

	if ax == None:
		fig, ax = plt.subplots()
	else:
		ax.clear()
	ax.plot(inside_edge_angles[:,0], inside_edge_angles[:,1], 'k-', alpha = .6)
	ax.plot(outside_edge_angles[:,0], outside_edge_angles[:,1], 'k-', alpha = .6)    
	plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
	plt.xlim(angles_limits_bottom[0],angles_limits_top[0])

	return(ax)

if __name__ == '__main__':

	
	"""
	

	# To project from world to screen,
	# first rotate and translate the object to the
	# to be "in front" of the camera (as done in earlier projection),
	# then multiply the resulting position vector with the matrix

	p_screen = cam_to_screen_homo(np.array([x_cam, z_cam]))

	# This results in homogeneous coorinates, where we have
	# to do the so called perspective divide (divide by the new z)
	# to get screen coordinates in pixels
	print("Using matrix stuff:", p_screen)

	# The whole computation can also be done explicitly without
	# the matrix stuff. Some could claim that this is easier,
	# and is certainly faster:
	screen = cam_to_screen([x_cam, z_cam])
	print("Direct computation:", screen)

	"""

	plot_perspective_view((-25,0), yaw = 0)
	plt.show()