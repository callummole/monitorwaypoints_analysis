import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import camera_models

"""
rootdir = "D:/Trout_rerun_gazevideos/Trout_508_1/000"

video_res = 1920, 1080
start_idx = 4500
"""
    #Rerun direct route works perfectly.


rootdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/Trout_8_1/000"
video_res = 1280, 720
start_idx = 50

    #Dataset 1 direct route crops the image

video_file ='world.mp4'

video = cv2.VideoCapture(os.path.join(rootdir,video_file))

video.set(cv2.CAP_PROP_POS_FRAMES, start_idx) #set to beginning frames.

print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_length = 15
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("sample_dataset1_2.mp4",fourcc, 30.0, video_res)

i =0
while i < frame_length:
    ret, oframe = video.read()          
    out.write(oframe)
    i += 1

out.release()
    

