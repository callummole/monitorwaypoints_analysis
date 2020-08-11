import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

EH = 1.2

H = np.linspace(-5, 20, 500)
V = np.linspace(-5, -1, 500)

#H, V = np.meshgrid(H, V)

hrad = (H*np.pi)/180
vrad = (V*np.pi)/180
    
zground = -EH/np.tan(vrad) 
xground = np.tan(hrad) * zground
lookahead = np.sqrt((xground**2)+(zground**2)) #lookahead distance from midline, before any rotation. Will be in metres.
TH = lookahead / 8.0
    
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(H, V, lookahead, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# plt.show()
def addintersectinglines(V, TH, vs = -2, col = 'b', al = .8):

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    xmin, xmax = ax.get_xlim()
    xrange = xmax - xmin
    for v in list(vs):
        idx = np.argmin( abs(V - v))
        th = TH[idx]
        print(th)
        ax.axhline(th, xmin = 0, xmax = 1 - (xmax - v) / xrange, linestyle = '--', color = col, alpha = al)
        ax.axvline(v, ymin = 0, ymax = 1- (ymax - th) / yrange, linestyle = '--', color = col, alpha = al)

#th = 2.5
#median vangle of manual is -3.88
##median vangle of auto  is -3.33
##median vangle of stock is -3.18
#v = (V[np.argmin(abs(TH-th))])
v = -3.88
margin = 1
vs = [v-margin, v, v+margin]

fig = plt.figure(1)
plt.plot(V, TH)
plt.xlabel("Vertical Angle (degrees)")
plt.ylabel("Time Headway (s)")
addintersectinglines(V,TH,vs)
v = -3.33
vs = [v-margin, v, v+margin]
addintersectinglines(V,TH,vs, col = 'r', al = .3)
plt.savefig('projection_error.png', format='png', dpi=800, bbox_inches = "tight", facecolor=plt.gcf().get_facecolor(), edgecolor='none')
plt.show()