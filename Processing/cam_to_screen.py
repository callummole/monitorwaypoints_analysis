import numpy as np

# The cave and screen spec
width = 1920
height = 1080
z = 1
x = 1.985
y = 1.115

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

print("The camera matrix is:")
print(M)

# A point's coordinates relative to the camera
x_cam = 1
y_cam = 1
z_cam = 5

print("Projecting world point", x_cam, y_cam, z_cam)


# To project from world to screen,
# first rotate and translate the object to the
# to be "in front" of the camera (as done in earlier projection),
# then multiply the resulting position vector with the matrix


p_homo = np.dot(M, [x_cam, y_cam, z_cam])

# This results in homogeneous coorinates, where we have
# to do the so called perspective divide (divide by the new z)
# to get screen coordinates in pixels
p_screen = p_homo[:2]/p_homo[-1]
print("Using matrix stuff:", p_screen)

# The whole computation can also be done explicitly without
# the matrix stuff. Some could claim that this is easier,
# and is certainly faster:
x_screen = fx*x_cam/z_cam + cx
y_screen = fy*y_cam/z_cam + cy
print("Direct computation:", x_screen, y_screen)

# But the matrix formulation comes often handy when you can
# precompute the whole transformation+projection and do everything
# in one go. And you don't go crazy with different coordinate systems.
# The End.
