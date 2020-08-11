import feather
import os
import pandas as pd
 
    


def main(datafilepath, filename):

    """loads csv and saves it to feather format"""

    datafolder = os.path.split(datafilepath)[0]  
    
    #steergaze_df = pd.read_csv(datafilepath, sep=',',header=0)           
    steergaze_df = pd.read_feather(datafilepath)           
    #if not 'feather' in filename:
    #    filename = filename + '.feather'

    #featherpath = datafolder + '/' + filename
    
    #feather.write_dataframe(steergaze_df, featherpath)    

    steergaze_df.to_csv(datafolder + '/' + filename)

if __name__ == '__main__':

    #datafilepath = "C:/git_repos/Trout18_Analysis/Post-processing/GazeAndSteering_newlatency_tidied_3D_remapped_addsample_improvetimeheadway.csv"
    #datafilepath = '../Data/trout_gazeandsteering_101019_addsample.csv'

    #datafilepath = '../Data/trout_twodatasets_full_2.csv'
    datafilepath = '../Data/trout_subset_3.feather'
    main(datafilepath, 'dataset2_subset.csv')
