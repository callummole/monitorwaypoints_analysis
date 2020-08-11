import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




#df = pd.read_csv('../Data/trout_twodatasets_full.csv')
#print(list(df.columns))

#hehe

df = pd.read_feather('../Data/trout_subset_2.feather')

#query_string = "dataset == 2 & T0_on_screen == 0 & roadsection < 2"
query_string = "dataset == 1"
#query_string = "dataset == 2 & T0_on_screen"
#query_string = "drivingmode < 3 & roadsection < 2 & T0_on_screen ==0 & confidence > .6 & on_srf == True"
df = df.query(query_string).copy()





#from 10 onwards there is a double timestamp.
for g, d in df.groupby(['ID']):
   
   i = 0
   for g2, d2 in d.groupby(['trialcode']):
   
      if i == 1: continue
      print(str(g) + '   :    ' + str(g2))
      print("data length: ", print(len(d2.index)) )
      #plt.plot(d2.currtime.values, d2.vangle.values)
      #plt.title(str(g) + '   :    ' + str(g2))
      #plt.show()
      i += 1


#totalrows = len(d2.midline_vangle_dist.values)

#exclude = "confidence > .6 & on_srf == True"
#d3 = d2.query(exclude).copy()
#dists = np.sqrt(d3.midline_vangle_dist.values ** 2 + d3.midline_hangle_dist.values ** 2)
#nwithin = len(dists[dists<20])
#print(1 - nwithin/totalrows)

#last_pos = []

#for g, d in df.groupby(['ID','block']):
#% FALL WITHIN X DEGREES OF MIDLINE
"""
dists = np.sqrt(df.midline_vangle_dist.values ** 2 + df.midline_hangle_dist.values ** 2)

total = len(dists)
print(total)
plt.figure()
for crit in np.linspace(0,20, num = 100):

    nwithin = len(dists[dists<=crit])
    perc = nwithin/total
   # print("Crit: ", crit)
   # print("Perc gaze: ", perc)
    plt.scatter(crit, perc)

plt.show()
"""

"""
#COMPARE MIDLINE DISTANCE VALUES
plt.figure()
for g, d in df.groupby(['dataset']):
   
   plt.scatter(d.midline_hangle_dist.values, d.midline_vangle_dist.values, alpha = .01)

plt.show()

"""


"""
#TRIAL LENGTH
trial_lengths = []
for g, d in df.groupby(['trialcode']):
    
    
    d.sort_values(['currtime'], inplace=True)	
    end = d.currtime.values[-1]
    start = d.currtime.values[0]

    trial_lengths.append(end-start)
#    print(g)
#    print(start-end, " seconds")
#    print((start-end) / 60, " minutes")

mn = np.mean(trial_lengths)
sd = np.sqrt(np.var(trial_lengths))
print("trial_length mean: ", mn)
print("trial_length sd: ", sd)
"""

"""
#BLOCK LENGTH
block_lengths = []
for g, d in d2.groupby(['ID','block']):
    
    
    d.sort_values(['currtime'], inplace=True)	
    end = d.currtime.values[-1]
    start = d.currtime.values[0]

    block_lengths.append( (end-start ) / 60.0)
#    print(g)
#    print(start-end, " seconds")
#    print((start-end) / 60, " minutes")

mn = np.mean(block_lengths)
sd = np.sqrt(np.var(block_lengths))
print("block_length mean: ", mn)
print("block_length sd: ", sd)

"""
"""
#NUMBER OF TRIALS
for g, d in df.groupby(['ID','drivingmode']):

    print(g)
    codes = set(d.trialcode.values)
    print("number of trials", len(codes))
"""
    
  #  posx = d.posx_mirror.values[-1]
  #  posz = d.posz_mirror.values[-1]

   # last_pos.append([posx, posz])

#last_pos = np.array(last_pos)

#print(np.mean(last_pos, axis = 0))


#existing_trials = set(df.trialcode.values)
#print(existing_trials)

#print(df[:10])

#data = df[:100]

#plt.plot(data.currtime.values, data.hangle.values)
#plt.plot(data.currtime.values, data.confidence.values * 60)
#plt.show()



