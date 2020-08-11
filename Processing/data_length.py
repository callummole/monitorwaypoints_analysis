import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

"""
gazefolderdir = "E:/EyeTrike_Backup/Recordings/Trout/ExperimentProper/"

      
file = 'gaze_on_surface_2019-12-05.csv'

data = pd.DataFrame(columns = ['ID','block','length'])
i = 0
for dirs in os.walk(gazefolderdir):        
        
    path = str(dirs[0]) + "/"
    print(path)
    
    folder_skip = "2B_reviewed"
    if folder_skip in path: continue

    if os.path.exists(path + file):        
        splitpath = path.split('/')
        print(splitpath[-2])
        exp, pp, block = splitpath[-2].split('_')  

        #if pp in ['7','11']:

        d = pd.read_csv(path + file)
        length =len(d.world_timestamp.values)

         #   plt.plot(d.viz_timestamp.values, d.x_norm.values)
          #  plt.show()
        
        data.loc[i] = [int(pp), block, length]
        i += 1
        
data.sort_values(['ID'], inplace=True)
for g,d in data.groupby(['ID']):plt.plot(d.length.values, d.ID.values, 'o', alpha = .5)
plt.show()
"""
    
    






df = pd.read_feather('../Data/trout_subset_old.feather')
#df = pd.read_csv('../Data/trout18_gazeandsteering_fake__3.csv')
#df = pd.read_csv('../Data/trout18_gazeandsteering_2019-12-06_1.csv')
#print(list(df.columns))
#query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
#query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True & dataset == 2"
#df = df.query(query_string).copy()
#df['dataset'] = 1
#last_pos = []
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



  #  posx = d.posx_mirror.values[-1]
  #  posz = d.posz_mirror.values[-1]