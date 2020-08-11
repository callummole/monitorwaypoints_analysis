#file to play around with loading the dump.
    
import gzip
import json
import numpy as np
import os
import msgpack


if __name__ == '__main__':
        
    #rootdir = "E:\\Trout_rerun_pupildumps"    
    rootdir = "E:\\Orca19_pilot_noload_block1\\venlab_machine"
        
    #fn = rootdir + "\\Trout_203_2.pupil.jsons.gz"
    fn = rootdir + "\\Orca19_None_Calibration_caltest.pupil.jsons.gz"   

    pupil_log = map(json.loads, gzip.open(fn))

    for topic, *_ in pupil_log:
        if topic == "notify.recording.started":
            break
    
    lines = iter(pupil_log)
    #json.dumps((topic, timestamper(), msgpack.loads(msg)))
    pupils = [],[]

    gaze_data = []
    timestamps = []

    
    for line in lines:
        print (line)
        
        topic, ts, data = line
        
        timestamps.append(ts)
        data['recv_ts'] = ts
        if topic == "pupil.0":
            pupils[0].append(data)
        if topic == "pupil.1":
            pupils[1].append(data)
        
        
        gaze_data.append(data)

        

    print("length: ", timestamps[-1] - timestamps[0])

    num = 2000
    plt.figure(2)

    pupils = pupils[0][:num], pupils[1][:num]

    for pupil in pupils:
        for p in pupil:
            
            npos= p['norm_pos']

            plt.plot(npos[0],npos[1], 'b.', alpha = .4)
     
    plt.show()