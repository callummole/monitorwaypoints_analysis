#file to play around with pldata.

import sys, os
from file_methods import *
import numpy

import matplotlib.pyplot as plt

def load_msgpack_gaze(file):
    
    gaze_data = []
    with open(file, "rb") as fh:
        for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):

            #each row is a packed dict containing topic, norm_pos, conf, ts
            data = msgpack.unpackb(payload, raw=False)                        
            gaze_data.append(data)

    return gaze_data


def plot_norm_pos(data):

    norm_pos = []
    plt.figure(2)
    
    data = data[:10000] #pick a slice.
    

    for d in data:
        
        np = d['norm_pos']

        plt.plot(np[0],np[1], 'b.', alpha = .4)
     
    plt.show()



if __name__ == '__main__':
    
    
    rootdir = "E:\\Trout_rerun_gazevideos"
    for dirs in os.walk(rootdir):

        path = str(dirs[0]) + "/"
        print (path)
        if os.path.exists(path + 'gaze.pldata'):

            #jami's pldata files are constructed a little differently. 
            gaze_data = []

            msgpack_file = os.path.join(path, 'gaze.recalib.pldata')
            gaze_data = load_msgpack_gaze(msgpack_file)
            
            plot_norm_pos(gaze_data)


            """
            normalised position tends to be from .2 to -.6 on the y-axis, and -.6 to .6 on the x axis.     

            The centre of the screen seems to be 0,0. Whereas really it should be .5,.5

            """
        

            
            