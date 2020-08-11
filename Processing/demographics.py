import feather
import os
import pandas as pd
import numpy as np
 
    


def main(datafilepath):

    """loads csv and saves it to feather format"""

    datafolder = os.path.split(datafilepath)[0]  
    
    columns = ['ID','Age','Gender','LicenseMonths','dataset']
    steergaze_df = pd.read_feather(datafilepath, columns = columns)

    #'ID', 'Age','Gender','LicenseMonths'
    print(list(steergaze_df.columns))

    for g, d in steergaze_df.groupby(['dataset']):

        print("\nDATASET: ", g)        

        n = len(set(d.ID.values))
        print("N:", n)
        dmg = np.empty([n, 3])

        for i, (pp, dd) in enumerate(d.groupby(['ID'])):
            dmg[i] = [dd.Age.values[0], dd.Gender.values[0], dd.LicenseMonths.values[0]]
            
        age = np.mean(dmg[:,0])
        print("mean age:", age)

        min_age = np.min(dmg[:,0])
        print("min age:", min_age)

        max_age = np.max(dmg[:,0])
        print("max age:", max_age)

        r_age = max(dmg[:,0]) - min(dmg[:,0])
        print("age range:", r_age)

        nm = len(dmg[dmg[:,1] == 2, 1])
        print("number of males:", nm)

        nf = len(dmg[dmg[:,1] == 1, 1])
        print("number of females:", nf)

        no = len(dmg[dmg[:,1] == 3, 1])
        print("number of others:", no)

        npr = len(dmg[dmg[:,1] == 4, 1])
        print("number of prefer not to say:", npr)

        lm = np.mean(dmg[:,2])
        print("mean license months:", lm)

        sdlm = np.sqrt(np.var(dmg[:,2]))
        print("sd license months:", sdlm)
    



if __name__ == '__main__':

    #datafilepath = "C:/git_repos/Trout18_Analysis/Post-processing/GazeAndSteering_newlatency_tidied_3D_remapped_addsample_improvetimeheadway.csv"
    #datafilepath = '../Data/trout_gazeandsteering_101019_addsample.csv'

    datafilepath = '../Data/trout_6.feather'
    main(datafilepath)
