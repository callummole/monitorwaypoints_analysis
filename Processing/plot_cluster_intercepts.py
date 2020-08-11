import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import calhelper as ch
from matplotlib.lines import Line2D


def get_ml_point(y_int, ml, cumdist):    
	
	y_int *= 8 #into metres
	closest = np.argmin(abs(y_int - cumdist))
	ml_pt = ml[closest]
	return ml_pt

def load_data():
	
	res = pd.read_csv("linmix_d1_6.csv")	
	return res

def main():

	ml = ch.get_midline()
	cumdist = ch.get_cumdist(ml)    

	fig_cm = np.array([13,6])
	fig_inc = fig_cm /2.54
	print(fig_inc)
	fig, ax = plt.subplots(1,1, constrained_layout = True, figsize = fig_inc)
	
	data = load_data()
	ef = data.query("clust_n == 1")

	active_color = tuple(np.array([3, 200, 255]) / 255)
	replay_color = tuple(np.array([255, 196, 3]) / 255)
	#replay_color = '#ff7f0e'
	stock_color = tuple(np.array([255, 3, 121]) / 255)
	cmaps = {0: [active_color], 1: [replay_color], 2: [stock_color]}

	max_inv_spr = np.max(1/(ef.spread.values**2))
	ax.plot(ml[:,1], ml[:,0]+6, '-', color = (.8,.8,.8), zorder = -3)
	for (dm), d in ef.groupby(['drivingmode']):
		
		offset = dm*2
		ax.plot(ml[:,1], ml[:,0]+offset, '-', color = (.8,.8,.8), zorder = -3)
		ints = d.intercept.values
		spreads = d.spread.values**2

		inv_spreads = 1/spreads
		wtmn = np.average(ints, weights = inv_spreads)
		ml_pt = get_ml_point(wtmn, ml, cumdist)	
		ax.scatter(ml_pt[1],ml_pt[0]+6, color = cmaps[dm][0], alpha = .5, s = 100, zorder = -dm)
		#ax.plot(ml_pt[1],ml_pt[0]+6, marker = 'o', markerfacecolor = "w", color = cmaps[dm][0], alpha = .5, markersize = 10, zorder = -dm)

		#print(wtmn)

		for intc, spr in zip(ints, spreads):	
		#	print(intc)
		#	print(spr)
			ml_pt = get_ml_point(intc, ml, cumdist)	
		#	print(ml_pt)
			ax.scatter(ml_pt[1],ml_pt[0]+offset, color = cmaps[dm][0], alpha = np.clip((1/spr) / max_inv_spr, .075,1), s = 50 + (spr * 200))
		

	#ax.axis('equal')
	ax.set_xlim(55,75)
	ax.set_ylim(-15,-26)
	#ax.invert_yaxis()
	ax.set_xlabel('World Z (m)', fontsize = 10)
	ax.set_ylabel('World X (m)', fontsize = 10)

	#annotations
	grey = (.4,.4,.4)
	ax.plot([56,60],[-18.5, -18.5], color = grey)
	ax.plot([60,60],[-18.25, -18.75], color = grey)
	ax.plot([56,56],[-18.25, -18.75], color = grey)
	ax.annotate('0.5 s (4 m)', xy =(56.5, -18), fontsize = 8)

	ax.annotate('Weighted Means', xy=(69.25,-17), xytext=(64, -16), fontsize = 8,
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
	
	ax.annotate('Midline Reference', xy=(65,-24.5), xytext=(63, -25.5), fontsize = 8,
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

	legend1 = [Line2D([0], [0], marker='o', color = "w", markerfacecolor=cmaps[0][0], label='Manual',
						  alpha =1),
				Line2D([0], [0], marker='o', color = "w", markerfacecolor=cmaps[1][0], label='Replay',
						  alpha =1),
				Line2D([0], [0], marker='o', color = "w", markerfacecolor=cmaps[2][0], label='Stock',
						  alpha =1)]

	ax.legend(handles = legend1, loc = [.01,.02], fontsize = 8, frameon = False, labelspacing = .25)
	ax.axis("off")  

	plt.savefig('ef_intercepts.svg', format='svg', dpi=800, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')

	plt.show()


if __name__ == '__main__':
	
	main()