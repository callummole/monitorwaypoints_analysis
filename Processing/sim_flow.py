"""script to simulate and plot flow on a circular trajectory, in the lab"""

import numpy as np
import drivinglab_projection as dp
import matplotlib.pyplot as plt


def makepoints(num = 20):

    x_angles = np.linspace(angles_limits_bottom[0], angles_limits_top[0], num)
    z_angles = np.linspace(angles_limits_bottom[1], -.1, num)
    
    xp, zp = np.meshgrid(x_angles, z_angles)
    
    points = np.array([xp.ravel(), zp.ravel()]).T

    points += (2*(-.5 + np.random.rand(*points.shape)))
    #points = dp.world_to_angles_through_screen(points)
    xpog, zpog = dp.angles_to_world(points) 
    points = np.array([xpog, zpog]).T

#    points = dp.world_to_angles(points)

    return(points)
    
def limit_to_display():
    
    plt.ylim(angles_limits_bottom[1],angles_limits_top[1])
    plt.xlim(angles_limits_bottom[0],angles_limits_top[0])

angles_limits_bottom = dp.screen_to_angles([0,0])[0]
angles_limits_top = dp.screen_to_angles([1,1])[0]

points = makepoints()
#plt.plot(points[:,0], points[:,1], 'b.')
#plt.show()

vel = 8 #m/s
dt = 1/60 #frame rate
yr = .2 #in radians
shutter_time = .25
shutter_frames = int(shutter_time/dt)
print(shutter_frames)

eye_in_world = np.array([2,15]) #xz of eye-in-world
eye_angles = dp.world_to_angles(eye_in_world)


yaw = 0
angles_hist = np.zeros( (len(points), 2, shutter_frames))
for f in range(shutter_frames):

    yaw = yaw+ (yr*dt)
    x_change = vel * dt * np.sin(yaw)
    z_change = vel * dt * np.cos(yaw)

    points -= [x_change, z_change]
    angles = dp.world_to_angles(points)
    angles -= eye_angles
    
    angles_hist[:, :, f] = angles

limit_to_display()
for a in angles_hist: plt.plot(a[0,:], a[1,:], 'C0-')
plt.show()





