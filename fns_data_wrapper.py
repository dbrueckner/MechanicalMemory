#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:52:27 2020

@author: D.Brueckner
"""

import numpy as np

          
class StochasticTrajectoryDataShape(object):
 
    def __init__(self,X,Y,delta_t,L,a,CSI_crit=0,mode_sim=0,P=None,mode_multi=False,mode_multiplestats=True):
        
        if len(X.shape)==2:
            X = X[:,:,np.newaxis]

        self.N_part = X.shape[0]
        self.N_t = []
        for j in range(0,self.N_part):
            self.N_t.append(len([x for x in X[j,:,0] if not np.isnan(x)]))
        self.N_part_max = X.shape[0]
        self.N_t_max = X.shape[1]
        self.N_variables = X.shape[2]
        self.CSI_crit = CSI_crit
        
        self.L = L
        self.a = a
        self.length = 2*a+L
        
        self.dt = delta_t
        self.time = np.linspace(0,self.N_t_max-1,self.N_t_max)*self.dt
        self.t_max = self.N_t_max*self.dt

        if mode_multi:
            self.X = np.zeros(X.shape)
            self.X[:] = np.nan
            self.centers = np.arange(-100,100)*(L+a)
            for j in range(0,self.N_part):
                system_id_init = np.argmin(np.abs(X[j,0,0]-self.centers))
                self.X[j,:,0] = X[j,:,0]-self.centers[int(system_id_init)]
            self.X[:,:,1] = X[:,:,1]
        else:
            self.X = X
            
            
        self.V = np.diff(self.X,axis=1)/self.dt
        self.A = np.diff(self.V,axis=1)/self.dt

        if mode_sim:
            self.alpha = Y
            self.P = P

            self.state = np.zeros(self.alpha.shape)
            self.state[:] = np.nan
            for j in range(0,self.N_part):
                self.state[j,:self.N_t[j]] = self.alpha[j,:self.N_t[j]]<0 #0 if E, 1 if C

        else:
            if mode_multiplestats:
                self.cellshapeindex = Y[:,:,0]
                self.cellaspectratio = Y[:,:,1]
                self.cellarea = Y[:,:,2]
            else:
                self.cellshapeindex = Y[:,:]

            cellarea_crit = 0.7*a**2
            self.state = np.zeros(self.cellshapeindex.shape)
            self.state[:] = np.nan
            for j in range(0,self.N_part):
                #self.state[j,:self.N_t[j]] = self.cellshapeindex[j,:self.N_t[j]]>self.CSI_crit #0 if E, 1 if C
                #self.state[j,:] = self.cellshapeindex[j,:]>self.CSI_crit #0 if E, 1 if C
                for t in range(0,self.N_t_max):
                    if np.isnan(self.X[j,t]):
                        self.state[j,t] = np.nan
                        """
                        elif np.abs(self.X[j,t])>L/2 and self.cellarea[j,t]>cellarea_crit: 
                            self.state[j,t] = 0 #elongated
                        elif np.abs(self.X[j,t])>L/2 and self.cellarea[j,t]<cellarea_crit: 
                            self.state[j,t] = 1 #compacted
                        """
                    elif np.abs(self.X[j,t])<L/2 and self.cellshapeindex[j,t]<CSI_crit:
                        self.state[j,t] = 0 #elongated
                    elif np.abs(self.X[j,t])<L/2 and self.cellshapeindex[j,t]>CSI_crit:
                        self.state[j,t] = 1 #compacted  
        
        self.state_bridge = np.zeros(self.state.shape)
        self.state_bridge[:] = np.nan
        for j in range(0,self.N_part):
            for t in range(0,self.N_t[j]):
                if mode_multi:
                    system_id = np.argmin(np.abs(X[j,t,0]-self.centers))
                    X_unitcell = X[j,t,0]-self.centers[int(system_id)]
                    if np.abs(X_unitcell)>self.L/2:
                        self.state_bridge[j,t] = 2 #on island
                    else:
                        self.state_bridge[j,t] = self.state[j,t] #on bridge 
                else:
                    if np.abs(self.X[j,t,0])>self.L/2:
                        self.state_bridge[j,t] = 2 #on island
                    else:
                        self.state_bridge[j,t] = self.state[j,t] #on bridge
                    

class StochasticTrajectoryData(object):
 
    def __init__(self,X,delta_t,mode_sim=0,P=None,mode_golgi=False):

        self.N_part = len([x for x in X[:,0,0] if not np.isnan(x)])
        self.N_t = []
        for j in range(0,self.N_part):
            self.N_t.append(len([x for x in X[j,:,0] if not np.isnan(x)]))
        self.N_part_max = X.shape[0]
        self.N_t_max = X.shape[1]
        self.N_variables = X.shape[2]
        
        self.dt = delta_t
        self.time = np.linspace(0,self.N_t_max-1,self.N_t_max)*self.dt
        self.t_max = self.N_t_max*self.dt

        self.X = X[:,:,0]
            
        self.V = np.diff(self.X,axis=1)/self.dt
        self.A = np.diff(self.V,axis=1)/self.dt

        if not mode_sim:
            self.X1 = X[:,:,1]
            self.X2 = X[:,:,2]
            
            self.L1 = np.abs(self.X - self.X1)
            self.L2 = np.abs(self.X - self.X2)
            
            self.length = self.L1 + self.L2
            self.alpha = np.abs( (self.L2-self.L1)/(self.L2+self.L1) )
            
            self.XL = np.nanmean(X[:,:,3])
            self.XR = np.nanmean(X[:,:,4])
            self.XBL = np.nanmean(X[:,:,5])
            self.XBR = np.nanmean(X[:,:,6])
            
            if mode_golgi:
                self.XGolgi = X[:,:,7]
                self.nuc_golgi = self.X-self.XGolgi
                
        
        if mode_sim:
            self.P = P
            
  
        
        
            