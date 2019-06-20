# Produce a graph where the markersize is scaled with the grid

import matplotlib.pyplot as plt

width = 40
height = 20

wid = 4
hei = wid * height / width
fig = plt.figure(1, figsize=(wid, hei))
sp = fig.add_subplot(111)
sp.set_xlim([0,width])
sp.set_ylim([0,height])
plt.axes().set_aspect('equal')

# line is in points: 72 points per inch
point_hei=hei*72

xval=[5,15,25,35,40]
yval=[1,2,18,7,18]
x1,x2,y1,y2 = plt.axis()

markersizescale = 1.5*72 * hei / height
scale = 1

plt.plot(xval, yval, marker='.', markersize=scale*markersizescale)
plt.grid('on', which='both')
plt.show()
