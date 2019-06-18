# Lerp experiments
'''
Here we went through a few linear extrapolation experiments
trying different norms (euclidean, manhattan, adjusted manhattan)
choosing elements seperately (diff in x, diff in y) - gave even better result
for squaring use x*x instead of x**2
for square rooting use (x)**.5 instead of np.sqrt(x)
then manhattan computation is slower than the euclidean equation

Lerp was a lie.  A lerp_vector could be precalcuated.
	lerp_vector = (old_loc - new_loc) / distance
	for speed in speeds:
		new_loc = old_loc + speed * lerp_vector
With this distance improvements make a minor difference.

TLDR:  lerp7 is the fastest, with no accuracy compromises.
'''

import time
import numpy as np

def rand():
    loc1 = np.random.rand(2)
    loc2 = np.random.rand(2)
    speed = np.random.rand()
    return loc1, loc2, speed

# Lerps
def lerp0(loc1, loc2, speed):
    # Frobenius Norm  (standard matrix norm, p=2-norm)
    distance = np.linalg.norm(loc1 - loc2)
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

def lerp1(loc1, loc2, speed):
    # Euclidean Norm  (standard vector norm, p=2-norm)
    distance = np.sqrt(sum((loc1 - loc2)**2))
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

def lerp2(loc1, loc2, speed):
    # Manhattan Norm  (semi-standard vector norm. p=1-norm)
    distance = sum(abs(loc1 - loc2))
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

sqrt2 = np.sqrt(2)
def lerp3(loc1, loc2, speed):
    # Adjusted Manhattan Norm
    distance = sum(abs(loc1 - loc2)) / sqrt2
    loc = loc2 + speed * (loc1 - loc2) / distance
    return loc

#sqrt2 = np.sqrt(2)
def lerp4(loc1, loc2, speed):
    # Adjusted Manhattan Norm with abbreviation
    loc = loc2 + speed * (loc1 - loc2) * sqrt2 / sum(abs(loc1 - loc2))
    return loc

#sqrt2 = np.sqrt(2)
def lerp5(loc1, loc2, speed):
    # Reciprocal of Adjusted Manhattan Norm
    reciprocal_distance = sqrt2 / sum(abs(loc1 - loc2))
    loc = loc2 + speed * (loc1 - loc2) * reciprocal_distance
    return loc

def lerp6(loc1, loc2, speed):
	# Elementwise Euclidean Norm by Keiran
	arr = loc1 - loc2
	distance = np.sqrt(arr[0]*arr[0] + arr[1]*arr[1])
	loc = loc2 + speed * (loc1 - loc2) / distance
	return loc

def lerp7(loc1, loc2, speed):
	# Elementwise Euclidean Norm
	x = loc1[0] - loc2[0]
	y = loc1[1] - loc2[1]
	distance =  (x*x + y*y)**.5
	loc = loc2 + speed * (loc1 - loc2) / distance
	return loc

def lerp8(loc1, loc2, speed):
	# Elementwise adjMan Norm
	x = loc1[0] - loc2[0]
	y = loc1[1] - loc2[1]
	distance =  abs(x) + abs(y)
	loc = loc2 + speed * (loc1 - loc2) * sqrt2 / distance
	return loc


# Lerps Profiling
if 1:
    n = int(1e5)

    # t = -time.time()
    # [lerp0(*rand()) for _ in range(n)]
    # t += time.time()
    # print(t/n)  # 10.671477317810058e-06
    # t = -time.time()
    # [lerp1(*rand()) for _ in range(n)]
    # t += time.time()
    # print(t/n)  # 10.013232231140136e-06
	#
    # t = -time.time()
    # [lerp2(*rand()) for _ in range(n)]
    # t += time.time()
    # print(t/n)  # 8.277862071990967e-06
	#
    # t = -time.time()
    # [lerp3(*rand()) for _ in range(n)]
    # t += time.time()
    # print(t/n)  # 8.4972882270813e-06
	#
    # t = -time.time()
    # [lerp4(*rand()) for _ in range(n)]
    # t += time.time()
    # print(t/n)  # 9.404857158660888e-06
	#
    # t = -time.time()
    # [lerp5(*rand()) for _ in range(n)]
    # t += time.time()
    # print(t/n)  # 8.168158531188965e-06
	#
    t = -time.time()
    [lerp6(*rand()) for _ in range(n)]
    t += time.time()
    print(t/n)  # 8.19807767868042e-06

    t = -time.time()
    [lerp7(*rand()) for _ in range(n)]
    t += time.time()
    print(t/n)  # 6.909528255462646e-06

    t = -time.time()
    [lerp8(*rand()) for _ in range(n)]
    t += time.time()
    print(t/n)  # 7.5494670867919925e-06

# Manhattan Adjustment
if 0:
    sqrt2 = np.sqrt(2)
    import matplotlib.pyplot as plt
    I = int(1e5)
    high, low = 100, -100
    x = (high-low) * np.random.random(size=(I, 2)) - low
    ds = []
    for i in range(len(x)-1):
        x0 = x[i]
        x1 = x[i+1]
        n1 = sum(abs(x0-x1))           # mathattan norm
        n1adj = n1 / sqrt2             # adjusted manhattan norm
        n2 = np.sqrt(sum((x0-x1)**2))  # euclidean norm
        d = n1adj - n2
        ds.append(d)
    std = .6827  # 1 standard deviation
    prop = sum(np.array(ds) < std/2 * -(high-low)/2) / I
    perc = 100*(1-prop)
    plt.hist(ds)
    plt.title('Demonstrates Adjusted Manhattan is a lower bound\n{:.1f}% fall within 1 standard deviation\n\nEuclidean vs Adjusted-Manhattan Norm Histogram'.format(perc))
    plt.xlabel('Method difference (n1adj - n2)')
    plt.ylabel('Samples {:,}'.format(I))
    plt.show()
