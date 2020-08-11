import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calhelper as ch




def load_data():

	df = pd.read_feather('../Data/trout_6.feather')
	query_string = "drivingmode < 3 & confidence > .8 & dataset == 2"
	df = df.query(query_string).copy()
	#correct for typo
	df.loc[df.ID == 203, 'ID'] = 503  

	return(df)

def get_point_cumdist(gm, tree, cumdist):    
	
	_, closest = tree.query(gm)
	gm_cumdist = cumdist[closest]

	return(gm_cumdist)

def cumulative_preview_yr(t, cumdist):
	""" takes a viewpoint and preview and returns the cumulative yawrate until preview """
	pass

def th_from_yrprev(yrprev):
	"""given a constant yrprev, what would be the predicted th"""


def main():

	"""
	if you treat cumulative yaw rate to gaze landing point as an index of 'difficulty ahead'. It makes sense that it's negatively related to TH.
	notes: 
	analysis has legs for trout paper 2.
	seems to be a power law around the sensitivity to cumulative yr preview, whereby small yr prev have a disproportionately large effect.

	"""


	df = load_data()

	#set up midline cumulative distance coordinate system
	ml = ch.get_midline()
	tree = ch.get_tree(ml)
	cumdist = ch.get_cumdist(ml)    

	for g, d in df.groupby(['trialcode']):

		#only include points where the yrp can be reliably calculated for
		ml_cumdist = d.midline_cumdist.values

		fp_x = d.midline_ref_world_x.values * d.startingposition.values
		fp_z = d.midline_ref_world_z.values * d.startingposition.values
		glps = np.array([fp_x, fp_z]).T
		cumdist_glps = get_point_cumdist(glps, tree, cumdist)		
		#print(cumdist_glps)
		last_cumdist = ml_cumdist[-1]
		#print(last_cumdist)
		keep_mask = cumdist_glps < last_cumdist
		plot_len = len(ml_cumdist[keep_mask])
		
		ml_cumdist /= 8.0
		traj = np.array([d.posx.values * d.startingposition.values, d.posz.values * d.startingposition.values]).T
			
		traj_tree = ch.get_tree(traj)
	



		yr = d.yawrate.values
		cum_yr = np.cumsum(abs(yr))	* 1/60	

		#closest indexes to gaze landing points
		_, closests_glp = traj_tree.query(glps)
		
		yrprev = [cum_yr[glp_i] - cyr for cyr, glp_i in zip(cum_yr, closests_glp)]
		
		th = d.th_along_midline.values

		min_yrp, max_yrp = min(yrprev), max(yrprev)

		#for a given point along the yrp_cum array, find out the index along the midline_cumdist that would result in yrp_const.
		yrp_const = np.median(yrprev)
		prev_th = []
		for cyr, mlcd in zip(cum_yr, ml_cumdist):
			#index of closest point to cyr + yrp_constant
			idx = np.argmin(np.abs(cum_yr - (cyr+yrp_const))) #minimum will be the closest point
			prev = ml_cumdist[idx] - mlcd
			prev_th.append(prev)

		fig, ax = plt.subplots(4,1, figsize = (10,8), sharex = True)

		th_max = max(prev_th)
		ax[0].plot(ml_cumdist[:plot_len], th[:plot_len], '-', alpha = .6)
		ax[0].plot(ml_cumdist[:plot_len], prev_th[:plot_len], '-', alpha = .6, color = 'm')

		ax[0].set_ylabel('Time Headway')
		ax[0].set_ylim(0, th_max)
		ax[1].plot(ml_cumdist[:plot_len], yr[:plot_len], 'o', alpha = .3, color = 'g')
		ax[1].set_ylabel('YawRate')
		ax[2].plot(ml_cumdist[:plot_len], yrprev[:plot_len], 'o', alpha = .3, color = 'red')
		ax[2].set_ylabel('YR prev')		
		ax[3].plot(ml_cumdist[:plot_len], prev_th[:plot_len], 'o', alpha = .3, color = 'm')
		ax[3].set_ylabel('TH given constant YRp')
		ax[3].set_ylim(0, th_max)

		plt.figure(2)
		plt.scatter(np.array(yrprev[:plot_len]), th[:plot_len], alpha = .1)
		plt.loglog()
		plt.show()


		
	
if __name__ == '__main__':
	
	main()