import pandas as pd 
import numpy as np 


#steergaze_df = pd.read_csv('../Data/trout_gazeandsteering_101019_addsample.csv')
steergaze_df = pd.read_feather('../Data/trout_steergaze.feather')
seg_df = pd.read_csv('../Data/segmentation_scaled_101019.csv')

master_seg = pd.DataFrame()

for block, blockdata in seg_df.groupby(['ID','block']):

    print(block)
    query_string = "ID == {} & block == {}".format(*block)    
    stgdata = steergaze_df.query(query_string).copy()
    stgdata.sort_values('currtime', inplace=True)
    seg_block = blockdata.copy()
    #seg_block.reset_index(inplace = True)
        
    for i, (index, row) in enumerate(seg_block.iterrows()):
        t = row['t1'], row['t2']
        launch_row = stgdata.loc[stgdata['currtime'] == t[0],:]
        land_row = stgdata.loc[stgdata['currtime'] == t[1],:]
#        print(t)
#        print("launch", launch_row)
#        print("land", land_row)

        if (land_row.empty) or (launch_row.empty): continue
        seg_block.loc[index, 'posx_1'] = launch_row['posx'].values[0]
        seg_block.loc[index, 'posx_2'] = land_row['posx'].values[0]
        seg_block.loc[index, 'posz_1'] = launch_row['posz'].values[0]
        seg_block.loc[index, 'posz_2'] = land_row['posz'].values[0]
        seg_block.loc[index, 'th_to_target_1'] = launch_row['th_to_target'].values[0]
        seg_block.loc[index, 'th_to_target_2'] = land_row['th_to_target'].values[0]

        #print(seg_block.loc[i, :])

    master_seg = pd.concat([master_seg, seg_block])

master_seg.to_csv('../Data/segmentation_scaled_101019.csv')