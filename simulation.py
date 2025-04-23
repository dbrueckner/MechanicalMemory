#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:15:08 2020

@author: D.Brueckner
"""

import numpy as np
import fns_simulation as fns_sim

delta_t = 3/60 
t_max = 20 #in hours

oversampling = 10
N_part = 20
N_t = int(t_max/delta_t)

L = 160
a = 40

a0 = 40
w = 6

beta = 1e-4
sigma = 100
n = 8
stiffness = 1

lambda_rate = 0.1
g0 = 1.85

alpha0 = 1
alpha_min = -2
gamma = 40


def sigmoid(x,k):
    return 1/(1+np.exp(-k*x))
def w_x(x,w,a,L):
    k = 0.2
    return ( (a-w)*( sigmoid(x-L/2,k)+ (1-sigmoid(x+L/2,k)) ) + w )
def g_x(x): #rate of C->E
    return lambda_rate*(w_x(x,w,a,L)/a0*g0)
def f_x(x): #rate of E->C
    result=lambda_rate*(1-w_x(x,w,a,L)/a0+w/a0)
    if result<0:
        result = 0
    return result

suffix = '_a' + str(a) + '_L' + str(L) + '_alpha0' + str(alpha0) + '_alphamin' + str(alpha_min) + '_lambda' + str(lambda_rate) + '_g' + str(g0)

params = (L,a,beta,sigma,n,stiffness,alpha0,alpha_min,gamma,f_x,g_x)

data = fns_sim.simulate(params,N_part,N_t,delta_t,oversampling)


import matplotlib.pyplot as plt
file_suffix = '.pdf'
plt.close("all")

import matplotlib as mpl
norm = mpl.colors.Normalize(0,2)
colors_CSI = [np.array([255,176,75])/255,
    np.array([169,94,255])/255]
colors = [[norm(0), colors_CSI[0]],
          [norm(1), colors_CSI[1]],
          [norm(2), "grey"],] #no data
cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors) 


H,W = 10,2
fig_size = [6,6]  
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

CSI_crit = 0.4

margin_left = 0.001
margin_right = 0.001
margin_bottom = 0.001
margin_top = 0.001
hspace = 0.001
wspace = 0.001

length = 2*a+L

fig = plt.figure()
chrt=0
fig.subplots_adjust(left=margin_left, bottom=margin_bottom, right=1-margin_right, top=1-margin_top, wspace=wspace, hspace=hspace)
for j in range(0,N_part):
    
    chrt+=1
    plt.subplot(H,W,chrt)
    plt.scatter(data.time,data.X[j,:],c=data.state_bridge[j,:],s=2,cmap=cmap,vmin=0,vmax=2)
    
    plt.xlim([0,data.t_max])
    plt.xticks([])

    plt.ylim([-length/2,length/2])
    plt.yticks([])
