# Machine Learning Python
'''
https://youtu.be/Q59X518JZHE
'''

# 1. Cleaning Data
# 2. Supervised/Unsupervised
# 3. Measure Performance / Test Algorithm

import matplotlib.pyplot as plt
import numpy as np

# Linear Regression
'''
from sklearn import datasets, linear_model

house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]

size2 = np.array(size).reshape((-1,1))

regr = linear_model.LinearRegression()
regr.fit(size2, house_price)
print(regr.coef_, regr.intercept_)

def graph(formular, x_range):
	x = np.array(x_range)
	y = eval(formular)
	plt.plot(x, y)
	return

graph('regr.coef_*x + regr.intercept_', range(1000, 2700))

plt.scatter(size, house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()
'''

# K-Means - Clustering
'''
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans


x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, .6, 11]
plt.scatter(x, y)
# plt.show()

X = np.array([x, y]).T

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
# print(centroids)
# print(labels)

colors = ['g.', 'r.', 'c.', 'y.']
for i in range(len(X)):
	print(X[i], labels[i])
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidth=5, zorder=10)
plt.show()
'''

#
'''
from copy import deepcopy
import pandas as pd
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

and the some description of KMeans
and csv required
'''
