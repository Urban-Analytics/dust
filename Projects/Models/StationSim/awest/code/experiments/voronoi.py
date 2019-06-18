import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='y', alpha=0.1)


# define a set of points on the sphere in any way you like
def convert_spherical_array_to_cartesian_array(spherical_coord_array, angle_measure='radians'):
	"""
	Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of
	the same shape in cartesian coordinate form (x,y,z). Based on the
	equations provided at:
	http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
	use radians for the angles by default, degrees if angle_measure == 'degrees'
	"""
	cartesian_coord_array = np.zeros(spherical_coord_array.shape)
	# convert to radians if degrees are used in input
	if angle_measure == 'degrees':
		spherical_coord_array[...,1] = np.deg2rad(spherical_coord_array[...,1])
		spherical_coord_array[...,2] = np.deg2rad(spherical_coord_array[...,2])
	# now the conversion to Cartesian coords
	cartesian_coord_array[...,0] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
	cartesian_coord_array[...,1] = spherical_coord_array[...,0] * np.sin(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
	cartesian_coord_array[...,2] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,2])
	return cartesian_coord_array

# generate random points on the unit sphere
def generate_random_array_spherical_generators(num_generators, sphere_radius, prng_object):
	"""
	Recoded using standard uniform selector over theta and acos phi,
	http://mathworld.wolfram.com/SpherePointPicking.html
	Same as in iPython notebook version
	"""
	u = prng_object.uniform(low=0,high=1,size=num_generators)
	v = prng_object.uniform(low=0,high=1,size=num_generators)
	theta_array = 2 * np.pi * u
	phi_array = np.arccos((2*v - 1.0))
	r_array = sphere_radius * np.ones((num_generators,))
	spherical_polar_data = np.column_stack((r_array,theta_array, phi_array))
	cartesian_random_points = convert_spherical_array_to_cartesian_array(spherical_polar_data)
	return cartesian_random_points

from scipy.spatial import SphericalVoronoi

points = generate_random_array_spherical_generators(50,1, np.random.RandomState(117))

fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='y', alpha=0.1)
ax.scatter(points[:,0], points[:,1], points[:,2])

radius = 1
center = np.array([0,0,0])
sv = SphericalVoronoi(points, radius, center)

fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='y', alpha=0.1)
ax.scatter(points[:,0], points[:,1], points[:,2])
ax.scatter(sv.vertices[:,0], sv.vertices[:,1], sv.vertices[:,2], color='r')


from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')
ax.plot_surface(x, y, z, rstride=5, cstride=5, color='y', alpha=0.1)
ax.scatter(points[:,0], points[:,1], points[:,2])

sv.sort_vertices_of_regions()
# this is not yet completely accurate
for n in range(0, len(sv.regions)):
	region = sv.regions[n]
	ax.scatter(points[n, 0], points[n, 1], points[n, 2], c='b')
	random_color = colors.rgb2hex(np.random.rand(3))
	polygon = Poly3DCollection([sv.vertices[region]], alpha=1.0)
	polygon.set_color(random_color)
	ax.add_collection3d(polygon)

plt.show()
