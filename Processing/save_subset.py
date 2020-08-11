import feather
import os
import pandas as pd
 
    


def main(datafilepath, filename):

    """loads csv and saves it to feather format"""

    datafolder = os.path.split(datafilepath)[0]  
    
    columns = ['vangle','hangle','yawrate','posx','posz','yaw','confidence','startingposition','currtime','on_srf','T0_on_screen','trialcode','condition','count',
    'midline_cumdist','th_along_midline','midline_vangle_dist','midline_hangle_dist','midline_ref_onscreen_x','midline_ref_onscreen_z','midline_ref_world_x','midline_ref_world_z',
    'ID','block','roadsection','drivingmode','dataset','currtimezero']    
    steergaze_df = pd.read_feather(datafilepath, columns = columns )               

    if not 'feather' in filename:
        filename = filename + '.feather'

    featherpath = datafolder + '/' + filename
    feather.write_dataframe(steergaze_df, featherpath)    



if __name__ == '__main__':

    #datafilepath = "C:/git_repos/Trout18_Analysis/Post-processing/GazeAndSteering_newlatency_tidied_3D_remapped_addsample_improvetimeheadway.csv"
    #datafilepath = '../Data/trout_gazeandsteering_101019_addsample.csv'

    datafilepath = '../Data/trout_6.feather'
    main(datafilepath, 'trout_subset_4')
