# Diffusion Flow
'''
import numpy as np
import matplotlib.pyplot as plt

def diffuse(u0, D=.01):
	# u0 is our array to be diffused by D
	# D is the scale at which we diffuse
	nx, ny = u0.shape
	u = np.empty(u0.shape)
	for i in range(1, nx-1):
		for j in range(1, ny-1):
			uxx = u0[i+1,j] - 2*u0[i,j] + u0[i-1,j]
			uyy = u0[i,j+1] - 2*u0[i,j] + u0[i,j-1]
			u[i,j] = u0[i,j] + D * (uxx + uyy)
	return u

n = 1000
ns = 1

World = np.zeros((n, n))
#i, j = np.random.randint(n, size=(2, ns))
World[n//2, n//2] = 1000


plt.contour(World)
plt.pause(.5)
for _ in range(100):
	World = diffuse(World)
	plt.imshow(World)
	plt.pause(.2)
'''

import numpy as np
import matplotlib.pyplot as plt

# plate size, mm
w = h = 100
# intervals in x-, y- directions, mm
dx = dy = 1
# Thermal diffusivity of steel, mm2.s-1
D = 4.
# Temperatures
Tcool, Thot = 300, 700


nx, ny = int(w/dx), int(h/dy)
dx2, dy2 = dx**2, dy**2
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))  # stable time step

u0 = Tcool * np.ones((nx, ny))
u = np.empty(u0.shape)

# Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
r2, cx, cy = 2**2, w//2, h//2
for i in range(nx):
	for j in range(ny):
		p2 = (i*dx-cx)**2 + (j*dy-cy)**2
		if p2 < r2:
			u0[i,j] = Thot

def do_timestep(u0, u):
	# Propagate with forward-difference in time, central-difference in space
	u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
		  (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
		  + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

	u0 = u.copy()
	return u0, u

# Number of timesteps
nsteps = 101
# Output 4 figures at these timesteps
mfig = [0, 10, 50, 100]
fignum = 0
fig = plt.figure()
for m in range(nsteps):
	u0, u = do_timestep(u0, u)
	if m in mfig:
		fignum += 1
		print(m, fignum)
		ax = fig.add_subplot(220 + fignum)
		im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
		ax.set_axis_off()
		ax.set_title('{:.1f} ms'.format(m*dt*1000))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()
