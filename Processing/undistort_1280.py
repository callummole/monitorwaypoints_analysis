import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import camera_models
import pickle

#DIRECT AND INDIRECT routes produce different outputs. This shouldn't happen.
rerun = True
video_file = 'world.mp4'

def load_pickled_camera(file):

    camera = pickle.load(open(file, 'rb'), encoding='bytes')
    image_resolution = camera[b'resolution']
    
    if b'rect_map' not in camera:
        camera_matrix = camera[b'camera_matrix']
        camera_distortion = camera[b'dist_coefs']
        rect_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, image_resolution, 0.0)
        rmap = cv2.initUndistortRectifyMap(
            camera_matrix, camera_distortion, None, rect_camera_matrix, image_resolution,
            cv2.CV_32FC1)
    else:
        rmap = camera[b'rect_map']
        rect_camera_matrix = camera[b'rect_camera_matrix']


    K, D, resolution, cm = camera[b'camera_matrix'], camera[b'dist_coefs'], camera[b'resolution'], rect_camera_matrix


    camera = {}
    camera['camera_matrix'] = rect_camera_matrix
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
    camera['resolution'] = image_resolution
    
    return(camera, rmap)


if rerun:
    #rootdir = "D:/Trout_rerun_gazevideos/Trout_502_1/000"
    path = "sample_rerun_2.mp4"
    video_res = 1920, 1080
    #start_idx = 4500
    start_idx = 0

    camera_spec = camera_models.load_intrinsics(directory="", cam_name='Pupil Cam1 ID2', resolution=video_res)   
    camera = {}
    camera['camera_matrix'] = camera_spec.K
    camera['dist_coefs'] = np.array([[.0,.0,.0,.0,.0]])
    camera['resolution'] = video_res
    #Rerun direct route works perfectly.
else:
    path = "sample_dataset1_2.mp4"
    video_res = 1280, 720
    start_idx = 0
    pickled_file = 'sdcalib.rmap.full.camera_unix.pickle'
    camera, rmap = load_pickled_camera(pickled_file)
    #Dataset 1 direct route crops the image

#video = cv2.VideoCapture(os.path.join(rootdir,video_file))
video = cv2.VideoCapture(path)

video.set(cv2.CAP_PROP_POS_FRAMES, start_idx) #set to beginning frames.

print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))



    #https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#undistort
    #According to pupil-labs you can just use camera_spec.undistort.
    #BUT this crops the image
while True:
    ret, oframe = video.read()      
    print(oframe.shape)
    if rerun:
        frame = camera_spec.undistort(oframe)        
    else:
        frame = cv2.remap(oframe, rmap[0], rmap[1], cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    plt.imshow(frame, cmap='gray')
    plt.pause(.2)
    plt.cla()
    
    plt.show()

