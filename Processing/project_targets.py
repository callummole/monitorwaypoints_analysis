import os
import numpy as np
import pandas as pd

def bring_to_camera(x, y, z, heading, pitch, px, py, pz):
    # Move the point to the camera coordinates
    x = px - x
    y = py - y
    z = pz - z
    # Rotate the point according to heading. The coordinate
    # system is apparently so that zero heading is the positive
    # z axis and the angle runs clockwise
    s = np.sin(-heading)
    c = np.cos(-heading)
    z, x = (
        z*c - x*s,
        z*s + x*c
            )

    # And pitch
    s = np.sin(-pitch)
    c = np.cos(-pitch)
    z, y = (
        z*c - y*s,
        z*s + y*c
            )
    return x, y, z

def project_point(x, y, z, heading, pitch, px, py, pz):
    # Move the point to the camera coordinates
    x = px - x
    y = py - y
    z = pz - z
    # Rotate the point according to heading. The coordinate
    # system is apparently so that zero heading is the positive
    # z axis and the angle runs clockwise
    s = np.sin(-heading)
    c = np.cos(-heading)
    z, x = (
        z*c - x*s,
        z*s + x*c
            )

    # And pitch
    s = np.sin(-pitch)
    c = np.cos(-pitch)
    z, y = (
        z*c - y*s,
        z*s + y*c
            )

    # Compute angles to the point. A projection should also
    # suffice, but the gaze locations are in angles
    hangle = np.arctan2(x, z)
    vangle = np.arctan2(y, z)
    return hangle, vangle


#def target_projector(targets):
#    def project(cam_position, heading):
#        angles = []
#        for t in targets:
#            angles.append(project_point(cam_position, heading, t))
#    return project

#TROUT_TARGETS = pd.read_csv(
#        os.path.join(os.path.dirname(__file__), "Trout18_Analysis", "Post-processing", "TargetPositions.csv")
#        )

TROUT_TARGETS = pd.read_csv("../Data/TargetPositions.csv")
TROUT_TARGETS['ycentre'] = 0.0
COND_TARGETS = {
    c: TROUT_TARGETS[TROUT_TARGETS.condition == c][['xcentre', 'ycentre', 'zcentre']].values
    for c in TROUT_TARGETS.condition.unique()
    }

def trout_projector(x, y, heading, condition):
    targets = COND_TARGETS[condition]
    angles = []
    cam_position = np.array((x, 1.2, z))
    for t in targets:
        angles.append(project_point(cam_position, heading, t))
    return np.array(angles)
    
