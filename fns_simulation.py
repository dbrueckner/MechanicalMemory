#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:41:56 2020

@author: D.Brueckner
"""
import numpy as np
import fns_data_wrapper as fns_data_wrapper

def simulate(params,N_part,N_t,delta_t,oversampling):
    
    (L,a,beta,sigma,n,stiffness,alpha0,alpha_min,gamma,f_x,g_x) = params
    
    alpha_EC = [alpha0,alpha_min]  
    length = 2*a+L
    
    dt = delta_t/oversampling

    N_t_oversampling = int(oversampling*N_t)
    N_t_preequil = 2*N_t_oversampling
    N_t_tot = N_t_oversampling+N_t_preequil
    
    X = np.ones((N_part,N_t))
    P = np.zeros((N_part,N_t,2))
    A = np.zeros((N_part,N_t))

    for j in range(0,N_part):

        state=np.random.choice([0,1])   
        alpha_prev = alpha_EC[state]

        x_prev = np.array([-length/2 + np.random.rand()*(length/2),0])
        p_prev = np.zeros(2)

        x_next = np.zeros(2)
        p_next = np.zeros(2)
        count_t = 0
        for t in range(0,N_t_tot):
            
            F_boundaries = F_obstacle(x_prev,0,length/2,stiffness,n) + F_obstacle(x_prev,0,-length/2,stiffness,n)

            x_next[0] = x_prev[0] + dt*( p_prev[0] + F_boundaries[0] )

            if np.abs(x_prev[0])<L/2:
                p_next[0] = p_prev[0] + dt*( F_p(p_prev[0],alpha_prev,beta) + gamma*F_boundaries[0] ) + np.sqrt(dt)*sigma*np.random.normal()
                p_next[1] = 0
            else:
                p_next = p_prev + dt*( F_p(p_prev,alpha_prev,beta) + gamma*F_boundaries[0]*np.array([1,0]) ) + np.sqrt(dt)*sigma*np.random.normal(size=(2))
                    

            if state==0:
                P_trans = f_x(x_prev[0])*dt
            else:
                P_trans = g_x(x_prev[0])*dt
                    
            #make a transition with probability P_trans (use random number on inverval [0,1])
            if np.random.rand()<P_trans: 
                state = 1-state
            alpha_next = alpha_EC[state]

            x_prev = x_next
            p_prev = p_next
            alpha_prev = alpha_next
            
            if t > N_t_preequil and np.mod(t,oversampling)==0:
                X[j,count_t] = x_prev[0]
                P[j,count_t,:] = p_prev
                A[j,count_t] = alpha_prev
                count_t += 1
    
    data = fns_data_wrapper.StochasticTrajectoryDataShape(X,A,delta_t,L,a,mode_sim=True,P=P)
    
    return data


def F_p(p,alpha,beta): return p*(-alpha-beta*np.dot(p,p))

def confinement_potential(x,n): return -x**(n-1)

def F_obstacle(x,dim,position_obstacle_in,stiffness,n):
    if position_obstacle_in<0:
        position_obstacle = position_obstacle_in + 5
        condition = x[dim] >= position_obstacle
    else:
        position_obstacle = position_obstacle_in - 5
        condition = x[dim] <= position_obstacle
    
    if(condition):
        result = np.zeros(2)
    else:
        vec = np.zeros(2)
        vec[dim] = 1
        result = stiffness*confinement_potential(x[dim]-position_obstacle,n)*vec
    return result
