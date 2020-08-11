#file to play around with pldata.

import sys, os
from file_methods import *
import numpy

import matplotlib.pyplot as plt
from pprint import pprint

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
        plt.xlim(0,1)
        plt.ylim(0,1)
     
    plt.show()


def regression(data):
    """
    regress onto recalib file.

    """

    recv_ts = []
    pupil_ts = []
    for gaze in gaze_data:

        recv_ts.append(gaze['recv_ts'])
        pupil_ts.append(gaze['timestamp'])

    ten = int(len(pupil_ts) * .1)
    pupil_to_recv = np.polyfit(pupil_ts[ten:-ten], recv_ts[ten:-ten], 1)
    new_recv = np.poly1d(pupil_to_recv)(pupil_ts)

    return(new_recv)


if __name__ == '__main__':
    
    #  rootdir = "E:\\Trout_rerun_gazevideos"
    #rootdir = "F:/Edited_Trout_rerun_Gaze_Videos/Trout_502_1/000/offline_data/gaze-mappings"
    rootdir = "F:/Edited_Trout_rerun_Gaze_Videos/Pupil_redetected/Trout_501_1/000/offline_data"
    for dirs in os.walk(rootdir):

        path = str(dirs[0]) + "/"
        print (path)
        
        gaze_data = []

        #msgpack_file = os.path.join(path, 'offline_pupil.pldata')
     #   msgpack_file = os.path.join(path, 'Default_Gaze_Mapper-d82c07cd.pldata')
        msgpack_file = os.path.join(path, 'offline_pupil.recalib.pldata')

        print("here")
        
        gaze_data = load_msgpack_gaze(msgpack_file)

        #gaze_data is a list.
    #    pprint(gaze_data)
        
        #print("Type", type(gaze_data))
        #print(gaze_data)
        #ts = regression(gaze_data)
        #plt.plot(range(len(ts)), ts)
#        plt.show()

        plot_norm_pos(gaze_data)


        """
        normalised position tends to be from .2 to -.6 on the y-axis, and -.6 to .6 on the x axis.     

        The centre of the screen seems to be 0,0. Whereas really it should be .5,.5

        """
        

            
            