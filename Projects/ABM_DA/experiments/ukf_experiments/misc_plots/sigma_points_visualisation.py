#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:16:04 2020

@author: medrclaa

Plot for showing why brute force technique of pf cant be used.

final plot will have 3x2 panels
top 2 estimate of position and corresponding particles generated for pf/ukf
middle 2 will be particles being fired to random exit gates

bottom 2 will be pf retaining some particles by brute force 
some ukf particles will go the right way but will lose covariance structure.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
from stationsim_densities import exit_points_to_polys
import sys

sys.path.append("..")
from sensors import generate_Camera_Rect

np.random.seed(2)

width = 20
height = 20
boundary = generate_Camera_Rect(np.array([0, 0]), 
                                    np.array([0, height]),
                                    np.array([width, height]), 
                                    np.array([width, 0]))
    
exit_gates = np.array([[10, 20], [20, 10]])
buffer = 1
exit_polys = exit_points_to_polys(exit_gates, boundary, buffer)
exit_bounds = [poly.intersection(boundary.exterior) for poly in exit_polys]
exit_bounds = [np.vstack(bound.coords.xy) for bound in exit_bounds]

########
# build agent position and sigmas
########
agent_position = np.array([5., 5.])
new_agent_position = np.array([15, 10])
agent_cov = np.array([[2, 0.5], [0.5, 2]])

agent_sqrt_cov = np.linalg.cholesky(agent_cov)

sigmas = np.reshape(np.tile(agent_position, 5), (5, 2), order = "C")
sigmas[1:3,:]+=  agent_sqrt_cov
sigmas[3:,:] -= agent_sqrt_cov
sigmas = sigmas.T
sigmas_mean = np.mean(sigmas, 1)
sigmas_cov = np.cov(sigmas)

#move sigmas randomly

gate_1_choices = np.random.choice(np.arange(sigmas.shape[1]), np.arange(sigmas.shape[1]), False)
gate_2_choices = list(set(np.arange(sigmas.shape[1])) - set(gate_1_choices))
gate_1_sigmas = sigmas[:, gate_1_choices]
gate_2_sigmas = sigmas[:, gate_2_choices]
gate_1_sigmas[0, :] += 5 + np.random.normal(scale = 0.1, size = 1)
gate_1_sigmas[1, :] += 10 + np.random.normal(scale = 0.1, size = 1)
gate_2_sigmas[0, :] += 10 + np.random.normal(scale = 0.1, size = 1)
gate_2_sigmas[1, :] += 5 + np.random.normal(scale = 0.1, size = 1)

sigmas_mean_2 = np.mean(gate_2_sigmas, 1) 
sigmas_cov_2 = np.cov(gate_2_sigmas)

#########
# plot 
#########

def covariance_to_ellipse(mean, cov, alpha, color, linestyle = '-'):
    
    lambd, v = np.linalg.eig(agent_cov)
    angle = np.rad2deg(np.arccos(v[0,0]))
    ell = Ellipse(xy = (mean[0], mean[1]),
                  width = lambd[0]*2, height = lambd[1]*2,
                  angle = angle, linewidth =1, color = color,
                  linestyle = linestyle, alpha = alpha)
    ell.set_facecolor('none')
    return ell

f, axs = plt.subplots(3, 1, True, False, figsize = (6, 4))

for i in range(axs.shape[0]):
    axs[i].set_xlim([0, width])
    axs[i].set_ylim([0, height])
    axs[i].set_xticklabels([])
    axs[i].set_yticklabels([])

axs[1].plot(2.5, 2.5)

for ax in axs.ravel():
    for bound in exit_bounds:
        ax.plot(bound[0, :], bound[1, :], color = "k", linewidth = 5)


#first row of plots
mean_position = axs[0].scatter(agent_position[0], agent_position[1], color = "k", marker = "x")
ell = covariance_to_ellipse(agent_position, agent_cov, 0.5, 'k')
sigmas_ell = covariance_to_ellipse(sigmas_mean, sigmas_cov, 1, 'red', "--")
axs[0].add_artist(sigmas_ell)
axs[0].add_artist(ell)
sigmas = axs[0].scatter(sigmas[0, :], sigmas[1, :], color = "red", alpha = 0.4, s = 50,
                 edgecolor = 'k')

#second row of plots
ell = covariance_to_ellipse(new_agent_position, agent_cov, 0.5, 'k')
axs[1].add_artist(ell)
axs[1].scatter(gate_1_sigmas[0, :], gate_1_sigmas[1, :], 
                 color = "r", alpha = 0.4, s = 50, marker = "o")
axs[1].scatter(gate_2_sigmas[0, :], gate_2_sigmas[1, :], 
                  color = "red", alpha = 0.4, s = 50, marker = "o",
                  edgecolor = "k")
axs[1].plot(new_agent_position[0], new_agent_position[1], color = "k", marker = "x")
    

#bottom row of plots
new_ell = covariance_to_ellipse(new_agent_position, agent_cov, 0.5, 'k')
new_sigmas_ell = covariance_to_ellipse(sigmas_mean_2, sigmas_cov_2, 1.0, "red", "--")

ell2 = axs[2].add_artist(new_ell)
axs[2].plot(new_agent_position[0], new_agent_position[1], color = "k", marker = "x")
mean_estimate = axs[2].scatter(sigmas_mean_2[0], sigmas_mean_2[1], color = "red", marker = "x")
#axs[2].scatter(gate_2_sigmas[0, :], gate_2_sigmas[1, :],
#               color = "orange", alpha = 0.4, s = 50, marker = "o",
#               edgecolor = "k")
ell3 = axs[2].add_artist(new_sigmas_ell)

axs[0].text(19, 17, "(a)", fontsize = 15)
axs[1].text(19, 17, "(b)", fontsize = 15)
axs[2].text(19, 17, "(c)", fontsize = 15)
handles = [ell, ell3, sigmas, mean_position, mean_estimate]
labels = ["State Covariance", "Estimated State Covariance",
          "Sigma Points", "State Mean", "Estimated State Mean"]
axs[2].legend(handles, labels, loc = 3)
plt.tight_layout()
#plt.suptitle("PF vs. UKF for Unknown Exit Gates")
plt.savefig("ukf_sigma_points.png")

"""
ax = []
#ax1 = plt.subplot(321)
for i in range(6):
    ax.append(plt.subplot(3,2,i+1))
    
for axis in ax:
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xlim([0, width])
    axis.set_ylim([0, height])
    
    axis.plot(5,5)
    
"""

plt.subplots_adjust(wspace =0, hspace = 0)

"""
for item in exit_polys:
    item = np.array(item.exterior.coords.xy).T
    plt.plot(item[:, 0], item[:, 1], color = "k")
"""
        

    
    
    