import numpy as np
import pandas as pd
from datetime import datetime
import multicalibrate
import extract_markers_pldata
import create_gaze_csv_pldata
#import stitch_and_project_gazeandsteering
import stitch_and_map_gazeandsteering
import segmentation
import matplotlib.pyplot as plt

def check_data_length(df):
    data = pd.DataFrame(columns = ['dataset','ID','trialcode','length'])
    print(data)

    for i, (g, d) in enumerate(df.groupby(['dataset','ID','trialcode'])):

    #  print(g)
        #if g[1] < 10: continue
        length = len(d.ID.values)    
        #plt.plot(d.currtime.values, d.yawrate.values)
        #plt.show()
        data.loc[i ] = [d.dataset.values[0], d.ID.values[0], d.trialcode.values[0], length]

        if length > 2000:
            print(d.trialcode.values[0])

    print(data)

    for g,d in data.groupby(['ID']):plt.plot(d.length.values, d.ID.values, 'o', alpha = .5)
    plt.show()


def further_process(files):
    
    print("STARTED FURTHER PROCESSING")
    path1 = "../Data/" + files[0]
   # path2 = "../Data/" + files[1]
    
    df1 = pd.read_csv(path1)
    print("loaded")
    #df2 = pd.read_csv(path2)
    #print("loaded")

    df1['dataset'] = 1
    #df2['dataset'] = 2

    #master = pd.concat([df1, df2])
    master = df1
    
    #check_data_length(master)
    
    print("running save_th_frompath")
    outfile = '../Data/trout_twodatasets_full.csv'
    
    import save_TH_frompath
    df = save_TH_frompath.main(master)
    print("finished save_th_frompath")
    df.to_csv(outfile)    


    gazemodes = np.array([[-22.74,70.4],[25,50.39]])
    import th_to_focal_point

    #check_data_length(df)
    print("running th_to_focal_point")
    df = th_to_focal_point.main(df, gazemodes)
    print("finished th_to_focal_point")
    df.to_csv(outfile)    
    

    #check_data_length(df)

    #df = pd.read_csv(outfile)
    print("running save_midline_cumdist")
    import save_midline_cumdist
    df = save_midline_cumdist.main(df)
    print("finished save_midline_cumdist, now saving df")
    df.to_csv(outfile)      
    

    datafilepath = '../Data/trout_twodatasets_full.csv'
    print("running save to feather")
    import save_to_feather
    save_to_feather.main(datafilepath, 'trout_6')
    print("finished save to feather")

    datafilepath = '../Data/trout_6.feather'
    print("running save to feather")
    import save_subset
    save_subset.main(datafilepath, 'trout_subset_3')    
    print("finished save to feather")

    print("running linmix")
    import linmix
    linmix.main(plot=False, outfile = 'linmix_res.csv', datafile = '../Data/trout_subset_3.feather')
    print("finished linmix")




def main(recalibrate, extract_markers, build_gaze, stitch_gaze, segment_gaze, rerun):
    
    #STEP 1) First we recalibrate.    
    #this takes a pupildumps folder to read form, and an outdirectory where the gaze video folders are    
    #gazefolderdir = "C:/git_repos/sample_gaze_Trout/Trout_506_2"    

    #date = datetime.date(datetime.now())
    #date = '2020-01-14'
    date = '2019-12-22'

    if rerun:
        gazefolderdir = "F:/Edited_Trout_rerun_Gaze_Videos/"
        #gazefolderdir = "F:/Test/d2/"
        gazefile = "offline_pupil.recalib.pldata"
        steerfile = "D:/Trout_rerun_steering_pooled"
        outfile = "/trout_rerun_gazeandsteering_" + str(date) + ".csv"
        #savedir = "C:/git_repos/sample_gaze_Trout/"
        savedir = "C:/git_repos/Trout18_Analysis/Data"
        gazebuilddate = '2019-11-21'

    else:
        gazefolderdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/"
        #gazefolderdir = "F:/Test/d1/"
        gazefile = 'pupil_data'                
        steerfile = "D:/Trout18_SteeringData_Pooled"
        outfile = "/trout18_gazeandsteering_" + str(date) + ".csv"
        #savedir = "C:/git_repos/sample_gaze_Trout/"
        savedir = "C:/git_repos/Trout18_Analysis/Data"
        gazebuilddate = '2019-12-05'
    
    if recalibrate == True: 
        if not rerun: raise Warning("cannot recalibrate on first dataset")
        print("RECALIBRATION STARTED")
        multicalibrate.main(pupildumpsfolder, gazefolderdir, gazefile)
        print("RECALIBRATION FINISHED")

    #STEP 2) extract the undistorted marker positions and save in square_marker_cache
    if extract_markers == True: 
        print("EXTRACT MARKERS STARTED")
        extract_markers_pldata.main(gazefolderdir, rerun)
        print("EXTRACT MARKERS FINISHED")

    #STEP 3) project gaze onto surface using square_marker_cache, build gaze_on_surface csvs. Needs surface definitions path.
    surf = "C:/git_repos/Trout18_Analysis/Processing/surface_definitions"
    if build_gaze == True:
        print("BUILDING GAZE ON SURFACE CSVS STARTED")
        
        create_gaze_csv_pldata.main(gazefolderdir, surf, gazefile, steerfile, rerun)        
        print("BUILDING GAZE ON SURFACE CSVS FINISHED")


    """
    After building gaze on surface csvs it's vital to check the synchronisation using check_sync_by_plotting.py.

    check_sync_by_plotting.py overlays the gaze and steering data onto the video file. 
    If the target objects are outlined then the sync is pretty good. Due to differences in frame-rates it won't be exact.
    Manually edit the latency until the target objects line up with the video.

    You may want to compute one single latency for the whole participant, or separate latencies for participants, for use in stitch and map...
    """

    #STEP 4) collate the data into large csv and stitch steering and gaze together.
    #stitch_and_project_gazeandsteering is the script that calculates a bunch of gaze and steering measures (e.g. angles, screen_coords)    
    
    if stitch_gaze == True:
        #'LATENCY = .15 #for dataset2. #checked 04.09.19
        #check processing date
        stitch_and_map_gazeandsteering.main(gazefolderdir, steerfile, savedir, outfile, rerun = rerun, date = gazebuilddate)

    #STEP 5) Segmentation using Pekkanen & Lappi (2017).


    #datafile = savedir + outfile
    datafile = savedir + '/trout_twodatasets_full.csv'
    segdate = '2020-02-2020'
    if rerun: 
        segfile = savedir + '/segmentation_scaled_rerun_' + str(segdate) + '.csv'
    else:
        segfile = savedir + '/segmentation_scaled_' + str(segdate) + '.csv'
    if segment_gaze:
        print("running segmentation")
        segmentation.main(datafile, segfile)
        print("finished segmentation")

    return outfile

if __name__ == '__main__':

    
    """To begin from scratch on the raw video files, set all the stages below to True.

    recalibrate: performs offline recalibration, weighting the gaze signal based on calibrations performed before and after each block. 
    extract_markers: extract the undistorted marker positions from the video and save in square_marker_cache
    build_gaze: project gaze onto surface using square_marker_cache, build gaze_on_surface csvs. Needs surface definitions path that defines the markers used for the surface.
    stitch_gaze: collate the data into large csv and stitch steering and gaze together. stitch_and_project_gazeandsteering is the script that calculates a bunch of gaze and steering measures (e.g. angles, screen_coords)    
    segment_gaze: Segmentation using Jami's segmentation alg published in Pekkanen & Lappi (2017). We've adapted it for the current paradigm.

    Uncomment the further process function to add more variables to the dataframe used for estimating a robust time headway signal, and ultimately for fitting the mixed linear reg model.
    """

    recalibrate = False
    extract_markers = False
    build_gaze = False
    stitch_gaze = False
    segment_gaze = True
    #rerun = False #flag to control which dataset and scripts to call
    #for rerun in [False, True]:
    rerun = True

    outfile = main(recalibrate, extract_markers, build_gaze, stitch_gaze, segment_gaze, rerun)    
#    files = []
#    for rerun in [True, False]:
    ########outfile = main(recalibrate, extract_markers, build_gaze, stitch_gaze, segment_gaze, rerun)
 #   files.append(outfile)
    

    #files = ['trout18_gazeandsteering_2020-01-14.csv','trout_rerun_gazeandsteering_2019-12-22.csv']
    #further_process(outfile)